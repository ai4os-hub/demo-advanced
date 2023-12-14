@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/2.1.1']) _

def projectConfig

pipeline {
    agent {
        label 'docker'
    }

    stages {
        stage('AI4EOSC SQA baseline dynamic stages') {
            steps {
                script {
                    projectConfig = pipelineConfig(
                        configFile: '.sqa/ai4eosc.yml'
                    )
                    buildStages(projectConfig)
                }
            }
            post {
                cleanup {
                    cleanWs()
                }
            }
        }
    }
}
