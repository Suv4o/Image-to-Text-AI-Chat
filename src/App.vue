<script setup lang="ts">
//@ts-ignore
import { extractColors } from 'extract-colors'
import { GetColorName } from 'hex-color-to-color-name';
import { PromptTemplate } from 'langchain/prompts'
import { MultiPromptChain } from 'langchain/chains'
import { BufferWindowMemory } from 'langchain/memory'
import { OpenAI, OpenAIChat } from 'langchain/llms/openai'
import '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import loaderSvg from './assets/svgs/loader.svg'
import { pipeline } from '@xenova/transformers'
import { ref, reactive, nextTick } from 'vue'

interface Message {
  text: string
  type: string
}

const reader = new FileReader()
reader.onload = (event) => {
  if (event.target) {
    imageUrl.value = event.target.result as string
  }
}

const messages = reactive<Message[]>([])
const inputMessage = ref('')
const isLoading = ref(false)
const showLoadingIndicator = ref(false)
const openAiApiKey = ref('')
const imageUrl = ref('')
const imageRef = ref(null)
const inputFileRef = ref()
const messagesRef = ref<HTMLDivElement>()
const imageLabels = ref('')
const chatHistory = ref('')
const imageColors = ref('')

function reset() {
  messages.splice(0, messages.length)
  inputMessage.value = ''
  imageUrl.value = ''
  imageLabels.value = ''
  chatHistory.value = ''
}

async function sendChatMessage(event: Event) {
  event.preventDefault()
  try {
    isLoading.value = true
    if (!inputMessage.value) {
      isLoading.value = false
      return
    }
    const input = inputMessage.value.trim()
    addMessage(inputMessage.value, 'user')
    inputMessage.value = ''
    const { chat_history, text } = await setChat({
      inputMessageUser: input,
      historySummary: chatHistory.value
    })
    chatHistory.value = chat_history
    addMessage(text, 'assistant', messages.length - 1)
    isLoading.value = false
  } catch (error) {
    isLoading.value = false
    console.error(error)
  }
}

async function passStream(stream: string, type: string, messageNumber: number | null = null) {
  if (messageNumber != null) {
    messages[messageNumber] = {
      text: messages[messageNumber].text + stream,
      type
    }
    await nextTick()
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  }
}

async function addMessage(message: string, type: string, messageNumber: number | null = null) {
  if (messageNumber != null) {
    messages[messageNumber] = {
      text: message,
      type
    }
  } else {
    messages.push({
      text: message,
      type
    })
  }
  await nextTick()
  if (messagesRef.value) {
    messagesRef.value.scrollTop = messagesRef.value.scrollHeight
  }
}

function setUrlFromImage(file: File) {
  if (file instanceof File) {
    reader.readAsDataURL(file)
  }
}

async function onFileChange(file: File) {
  try {
    reset()
    if (!openAiApiKey.value) {
      alert('Please enter your OpenAI API key')
      const input = inputFileRef.value.$el.querySelector('input')
      input.value = ''
      return
    }
    setUrlFromImage(file)
    isLoading.value = true
    showLoadingIndicator.value = true
    await nextTick()
    await loadImageLabels()
    const { chat_history, text } = await setChat({
      inputMessageUser: 'Hello! I attached an image for you to assist me.',
      historySummary: ''
    })
    chatHistory.value = chat_history
    addMessage(text, 'assistant', messages.length - 1)
  } catch (error) {
    console.error(error)
  } finally {
    isLoading.value = false
    showLoadingIndicator.value = false
  }
}

async function setChat(
  msg = {
    inputMessageUser: '',
    historySummary: ''
  }
) {
  const memory = new BufferWindowMemory({
    k: 10,
    memoryKey: 'chat_history'
  })

  if (msg.historySummary) {
    await memory.saveContext(
      {
        input: msg.inputMessageUser
      },
      {
        output: msg.historySummary
      }
    )
  }

  const model = new OpenAIChat({
    openAIApiKey: openAiApiKey.value,
    modelName: 'gpt-3.5-turbo-0613',
    temperature: 1,
    streaming: true
  })

  const promptNames = ['content', 'general-description', 'instagram-capture', 'seo-description', 'color-palette']

  const promptDescriptions = [
    'Good for answering questions about what is in the image',
    'Good for writing general description of the image',
    'Good for writing instagram capture of the image',
    'Good for writing description of the image for visually impaired people or for SEO purposes',
    'Good for getting the color palette of the image'
  ]

  const promptContent =
    PromptTemplate.fromTemplate(`The following is a conversation between a human and an AI. Your role is to tell the human what is in the image the human attached. The attached image has been labeled with "${imageLabels.value}" to describe its content. These labels are sorted by relevance, with the first label having the highest probability and the last label having the lowest probability of being included in the image.
      If the human ask to describe what is in the image, you must provide a summary based on my knowledge of the labels provided. If the human ask if a certain subject or activity is in the image, you must answer with yes or no, and if you are not sure, you will indicate that based on your knowledge, you not sure if it is in the image.
      Please note that the human may provide additional corrections on what is in the image, and you must listen to and incorporate those suggestions as needed.
      You must not ask the human questions for more details about the desired description or the elements present in the image. If a human asks you to write a description, proceed without requesting additional information and use the labels provided below. It is important not to ask the human to provide you the labels or describe what is in the image.
      If the Human asks about the content and does not specify that it is for the attached image, you must assume that it is for the image above. Do not ask any additional questions. Simply write the description or caption using the labels.
      Current conversation:
      {chat_history}
      Human:
      {input}
      AI:`)

  const promptGeneralDescription =
    PromptTemplate.fromTemplate(`The following is a conversation between a human and an AI. Your role is to help the human write a description of an image. The human will be provide details on what the description is needed for, as well as possibly asking some questions or making corrections if the description is not suitable for their needs. Be helpful and assist as much as you can. You must not ask the human questions for more details about the desired description or the elements present in the image. If a human asks you to write a description, proceed without requesting additional information and use the labels provided below. It is important not to ask the human to provide you the labels or describe what is in the image. If you don't have an answer, simply reply that you cannot help. If the user asks questions beyond your role as an assistant to assist with writing image descriptions based on image content, simply state that as an assistant, you cannot help with that task.
      The attached image has the following labels: "${imageLabels.value}" that describe its content. The labels are sorted by relevance, with the first label having the highest probability and the last label having the lowest probability of being included in the image.
      If the Human asks to write a description or caption and does not specify that it is for the attached image, you must assume that it is for the image above. Do not ask any additional questions. Simply write the description or caption using the labels.
      Current conversation:
      {chat_history}
      Human:
      {input}
      AI:`)

  const promptInstagramDescription =
    PromptTemplate.fromTemplate(`The following is a conversation between a human and an AI. Your role is to assist the human write a a captivating Instagram caption for an image. The attached image has the following labels: "${imageLabels.value}" that describe its content. The labels are sorted by relevance, with the first label having the highest probability and the last label having the lowest probability of being included in the image. If the human asks for corrections, please reply with the necessary corrections. Adding emojis can add a personal touch to the capture. Adding Instagram hashtags at the end of the caption is also a good idea.
      You must not ask the human questions to provide labels of more details about the desired description or the elements present in the image. If a human asks you to write a description, proceed without requesting additional information and use the labels provided below. It is important not to ask the human to provide you the labels or describe what is in the image.
      If the Human asks to write a caption and does not specify that it is for the attached image, you must assume that it is for the image above. You must not ask any additional questions. You also must not ask specific details about the image. Simply write the description or caption using the labels for the image provided. The capture will be for the image above.
      Current conversation:
      {chat_history}
      Human:
      {input}
      AI:`)

  const promptSeo =
    PromptTemplate.fromTemplate(`The following is a conversation between a human and an AI. Your role is to assist the human to write alternation text for an image HTML <img/> alt attribute. The image has been attached and has been labeled with "${imageLabels.value}" to describe its content. These labels are sorted by relevance, with the first label having the highest probability and the last label having the lowest probability of being included in the image.
      Write a brief description that will be added in the alt tag to describe the image for visually impaired individuals, allowing screen readers to provide a description of the contents. This is also beneficial for SEO purposes when individuals search on the web.
      Please note that the human may provide additional corrections on what is in the image, and you must listen to and incorporate those suggestions as needed.
      You must not ask the human questions for more details about the desired description or the elements present in the image. If a human asks you to write a description, proceed without requesting additional information and use the labels provided below. It is important not to ask the human to provide you the labels or describe what is in the image.
      If the Human asks to write alternation description and does not specify that it is for the attached image, you must assume that it is for the image above. Do not ask any additional questions. Simply write the description or caption using the labels.
      Current conversation:
      {chat_history}
      Human:
      {input}
      AI:`)

  const promptColorPalette =
    PromptTemplate.fromTemplate(`The following is a conversation between a human and an AI. Your role is to assist the human to to tell the colors in that are included in the image using your own description. The image has been attached and its dominant colors are "${imageColors.value}.” Assist with a short summary with your imagination on how the colors blend into the image. As an assistant, you should not ask the human to identify the colors in an image. Your role is to assist the human without asking for any additional details.
      Current conversation:
      {chat_history}
      Human:
      {input}
      AI:`)

  const promptTemplates = [
    promptContent,
    promptGeneralDescription,
    promptInstagramDescription,
    promptSeo,
    promptColorPalette
  ]

  const chain = MultiPromptChain.fromLLMAndPrompts(model, {
    promptNames,
    promptDescriptions,
    promptTemplates,
    llmChainOpts: {
      memory
    }
  })

  let hasNewMessageBeenSet = false
  let startStreaming = false
  const streamArray: String[] = []

  const response = await chain.call(
    {
      input: msg.inputMessageUser
    },
    [
      {
        handleLLMNewToken(token: string) {
          streamArray.push(token)
          if (
            streamArray[streamArray.length - 1] === '' &&
            streamArray[streamArray.length - 2] === ''
          ) {
            startStreaming = true
          }
          if (startStreaming) {
            if (!hasNewMessageBeenSet) {
              addMessage(token, 'assistant')
              hasNewMessageBeenSet = true
            }
            passStream(token, 'assistant', messages.length - 1)
          }
        }
      }
    ]
  )

  return {
    ...(await memory.loadMemoryVariables({})),
    ...response
  }
}

async function loadImageLabels() {
  const imageToText = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning')
  const classifier = await pipeline(
    'zero-shot-image-classification',
    'Xenova/clip-vit-base-patch32'
  )

  // Colors in the image
  const extractedColors = await extractColors(imageUrl.value)
  imageColors.value = extractedColors.map((color: any) => {
    return GetColorName(color.hex)
  }).join(', ')
  

  // Image labels from GPT 2 Image Captioning
  const imageCapture = await imageToText(imageUrl.value)

  const model = new OpenAI({
    openAIApiKey: openAiApiKey.value,
    modelName: 'gpt-3.5-turbo-0613',
    temperature: 0
  })

  const labelsFromImageCapture = await model.call(
    `Extract only the nouns and verbs from the following text: "${imageCapture[0].generated_text}". Output them as a single array of strings in JavaScript format, such as ["noun1", "noun2", "verb1", "verb2"]. Only one array should be returned.`
  )

  const indexOfFirstBracketOfImageCaptureLabels = labelsFromImageCapture.indexOf('[')
  const indexOfLastBracketsOfImageCaptureLabels = labelsFromImageCapture.indexOf(']') + 1
  const onlyArrayImageCaptureLabels = labelsFromImageCapture.slice(
    indexOfFirstBracketOfImageCaptureLabels,
    indexOfLastBracketsOfImageCaptureLabels
  )

  const labelsCaptureXenova = JSON.parse(onlyArrayImageCaptureLabels)

  // Image labels from TensorFlow
  const labelsClassesTensorflow = [] as any[]
  const labelsObjectsTensorflow = [] as any[]
  const img = imageRef.value
  if (!img) return
  const modelTensorflowClassification = await mobilenet.load()
  const modelTensorflowObjectDetection = await cocoSsd.load()
  const predictionsTensorflowClassification = await modelTensorflowClassification.classify(img)
  const predictionsTensorflowObjectDetection = await modelTensorflowObjectDetection.detect(img)

  predictionsTensorflowClassification.forEach((prediction) => {
    labelsClassesTensorflow.push(...prediction.className.split(',').flat())
  })

  predictionsTensorflowObjectDetection.forEach((prediction) => {
    labelsObjectsTensorflow.push(...prediction.class.split(',').flat())
  })

  const predictedLabels = [
    ...new Set([...labelsCaptureXenova, ...labelsClassesTensorflow, ...labelsObjectsTensorflow])
  ]

  const scoresPredictions = await classifier(imageUrl.value, predictedLabels)
  const scoresPredictionsSorted = scoresPredictions.sort((a: any, b: any) => b.score - a.score)
  const scoresPredictionsSortedTop = scoresPredictionsSorted.slice(0, 4)

  const scoresPredictionsSortedTopString = scoresPredictionsSortedTop
    .map((prediction: any) => prediction.label)
    .join(', ')

  imageLabels.value = scoresPredictionsSortedTopString
}
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
    <div class="mx-auto max-w-3xl h-[100vh] flex flex-col justify-end py-4">
      <div ref="messagesRef" class="overflow-y-scroll">
        <p v-if="!imageUrl" class="text-2xl mb-8 text-blue-700">
          This is a friendly AI assistant to help you describe your image. To begin, simply add your
          Open AI api key at the bottom of the page and then attach your image. Once the image has
          been analysed, you can start talking with the assistant. You can ask the assistant to:
          <ul class="list-decimal my-3">
            <li>- Write a description for the image.</li>
            <li>- Write an Instagram caption for your image.</li>
            <li>- Write a description for your image's HTML tag alternate text.</li>
          </ul>
          Enjoy chatting!
        </p>
        <div class="mb-4 mx-1 mt-1 flex justify-center">
          <v-file-drop @change="onFileChange" ref="inputFileRef">
            <div
              class="flex justify-center items-center min-h-[200px] w-full rounded-xl shadow-lg ring-1 ring-blue-700 overflow-hidden"
            >
              <p v-if="!imageUrl" class="text-3xl text-blue-700 px-8 text-center">
                Drop your image here.
              </p>
              <img
                ref="imageRef"
                v-else
                :src="imageUrl"
                alt="Image to text"
                class="w-full h-full"
              />
            </div>
          </v-file-drop>
        </div>
        <div v-for="(message, index) in messages" :key="index" class="inset-x-0 pb-4 px-1">
          <div
            class="rounded-xl p-6 shadow-lg"
            :class="[
              message.type === 'user' && 'bg-blue-600',
              message.type === 'assistant' && 'bg-white ring-1 ring-blue-700'
            ]"
          >
            <p
              v-html="message.text.replace(/\n/g, '<br />').replace('<br />', '')"
              class="sm:text-xl leading-6 [&_a]:font-bold [&_a]:text-blue-800"
              :class="[
                message.type === 'user' && 'text-white',
                message.type === 'assistant' && 'text-blue-700'
              ]"
            ></p>
          </div>
        </div>
      </div>
      <div v-if="isLoading && showLoadingIndicator" class="flex justify-center mx-1 my-4">
        <img :src="loaderSvg" class="h-12 w-12" alt="loading" />
      </div>
      <form @submit="sendChatMessage" class="relative px-1 mb-8">
        <div
          class="overflow-hidden rounded-lg pb-12 shadow-sm ring-1 ring-inset ring-blue-700 focus-within:ring-2 focus-within:ring-blue-500"
          :class="[
            isLoading && 'ring-gray-300 focus-within:ring-2 focus-within:ring-gray-200',
            !imageUrl && 'ring-gray-300 focus-within:ring-2 focus-within:ring-gray-200'
          ]"
        >
          <textarea
            v-model="inputMessage"
            rows="2"
            name="comment"
            id="comment"
            :disabled="isLoading || !imageUrl"
            class="block w-full resize-none border-0 bg-transparent py-4 text-blue-700 placeholder:text-blue-300 focus:ring-0 sm:text-xl sm:leading-6"
            :class="[
              isLoading && 'cursor-not-allowed text-gray-300 placeholder:text-gray-200',
              !imageUrl && 'cursor-not-allowed text-gray-300 placeholder:text-gray-200'
            ]"
            placeholder="Add your question..."
          />
        </div>

        <div class="absolute inset-x-0 bottom-0 flex justify-between py-2 pl-3 pr-3">
          <div class="flex items-center space-x-5">
            <div class="flex items-center"></div>
            <div class="flex items-center"></div>
          </div>
          <button
            :disabled="isLoading || !imageUrl"
            type="submit"
            class="rounded-md bg-white px-2.5 py-1.5 sm:text-xl font-semibold text-blue-700 shadow-sm ring-1 ring-inset ring-blue-700 hover:bg-blue-500-50 inline-flex gap-2"
            :class="[
              isLoading && 'cursor-not-allowed text-gray-300 ring-gray-300',
              !imageUrl && 'cursor-not-allowed text-gray-300 ring-gray-300'
            ]"
          >
            Send
          </button>
        </div>
      </form>
      <p class="text-xl mb-2 mx-1 text-blue-700">Please add your OPEN AI API key below:</p>
      <div class="w-full px-1">
        <div>
          <input
            v-model="openAiApiKey"
            type="text"
            name="open_ai_api_key"
            class="block w-full rounded-md border-0 py-1.5 text-blue-700 shadow-sm ring-1 ring-inset ring-blue-700 placeholder:text-blue-300 focus:ring-2 focus:ring-inset focus:ring-blue-600 sm:text-xl sm:leading-6"
            placeholder="ADD YOUR OPEN AI API KEY"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.file-upload__container {
  @apply w-full;
}
</style>
