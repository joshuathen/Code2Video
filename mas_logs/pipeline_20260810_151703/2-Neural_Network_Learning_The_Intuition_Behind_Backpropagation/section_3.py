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
        self.setup_layout("Forward Pass: Making a Prediction", [
            "Data flows through neuron layers to make predictions.", 
            "Weighted sums and activation create the network's guess.", 
            "Example: A network classifying images as cats or dogs."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00AAFF"))
        
        input_vec = Arrow(start=LEFT*0.5, end=RIGHT*0.5, color="#00AAFF")
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        input_group = Group(input_vec, cat_icon).arrange(RIGHT)
        self.place_at_grid(input_group, 'C2', scale_factor=0.5)
        self.play(FadeIn(input_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        nodes = VGroup(*[Dot(color="#FFFF00", radius=0.15) for _ in range(3)]).arrange(DOWN)
        self.place_at_grid(nodes, 'C3', scale_factor=1.0)
        self.play(FadeIn(nodes))
        
        sum_text = MathTex(r"\\sum w \\cdot x + b", color="#FFFF00").scale(0.9)
        self.place_at_grid(sum_text, 'D3', scale_factor=0.9)
        self.play(Write(sum_text))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        result_text = Text("70% Cat, 30% Dog", font_size=24, color="#00FF00")
        dog_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg")
        result_group = VGroup(result_text, dog_icon).arrange(RIGHT)
        self.place_at_grid(result_group, 'E3', scale_factor=0.9)
        self.play(Write(result_group))
        
        self.wait(2)
