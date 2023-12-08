@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/2.1.1']) _

def projectConfig

pipeline {
    agent {
        label 'docker'
    }

    stages {
        stage('User pipeline job') {
            steps {
                script {
                    build(job: '/AI4OS-HUB-TEST/' + env.JOB_NAME.drop(10))
                }
            }
        }
        stage('SQA baseline dynamic stages') {
            steps {
                script {
                    projectConfig = pipelineConfig(
                        configFile: '.sqa/ai4eosc.yml',
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
