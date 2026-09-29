from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Transformer: The 'Attention' Mechanism", [
            "Self-attention acts like a dynamic spotlight.",
            "Every word weighs its focus on others.",
            "The mechanism uses Query and Key vectors.",
            "Softmax calculates these relationship weights.",
            "Meaning changes based on context weights."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Represent a spotlight [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg] moving across a sentence in #FFFF00.
        self.lecture[0].set_color("#FFFF00")
        spotlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg")
        self.place_at_grid(spotlight, 'B2', scale_factor=0.5)
        self.play(FadeIn(spotlight))
        self.play(spotlight.animate.move_to(self.grid['B5']), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        # Show lines connecting a central word to others with weights #33FF57.
        self.lecture[1].set_color("#33FF57")
        sentence = VGroup(*[Text(w, font_size=24) for w in ["Bank", "river", "money"]])
        sentence.arrange(RIGHT, buff=0.5)
        self.place_at_grid(sentence, 'D3', scale_factor=0.8)
        self.add(sentence)
        line = Line(sentence[0].get_top(), sentence[1].get_top(), color="#33FF57")
        self.play(Create(line))
        
        # === Animation for Lecture Line 3 ===
        # Visualize Query, Key, and Value vectors as distinct colored bars (#3357FF, #FF33A1, #FF9933).
        self.lecture[2].set_color(WHITE) # Reset others if needed
        q_vec = Rectangle(height=0.5, width=1.5, color="#3357FF", fill_opacity=1)
        k_vec = Rectangle(height=0.5, width=1.5, color="#FF33A1", fill_opacity=1)
        v_vec = Rectangle(height=0.5, width=1.5, color="#FF9933", fill_opacity=1)
        vecs = VGroup(q_vec, k_vec, v_vec).arrange(DOWN)
        self.place_at_grid(vecs, 'C5', scale_factor=0.6)
        self.play(FadeIn(vecs))
        
        # === Animation for Lecture Line 4 ===
        # Show the Softmax function mapping weights to a probability distribution.
        self.lecture[3].set_color("#FF33A1")
        softmax = Text("Softmax", font_size=24)
        self.place_at_grid(softmax, 'E3', scale_factor=0.8)
        self.play(Write(softmax))
        
        # === Animation for Lecture Line 5 ===
        # Update the word representation based on the calculated attention weights.
        self.lecture[4].set_color("#FF9933")
        self.play(sentence[0].animate.set_color("#FF9933"))
        self.wait(1)
