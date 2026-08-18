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
        lecture_lines = [
            "Weights define the MLP's fact templates.",
            "Layer one acts as a pattern matcher.",
            "Layer two functions as the fact generator.",
            "A vector input triggers a specific neuron spike.",
            "The neuron projects the fact to the output."
        ]
        self.setup_layout("The Mechanics: Weights as Fact Templates", lecture_lines)
        
        # Pre-construct visuals
        neuron_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        
        neuron1 = SVGMobject(neuron_asset, color="#FFFF00").scale(0.5)
        neuron2 = SVGMobject(neuron_asset, color="#FFFFFF").scale(0.5)
        
        connection = Line(ORIGIN, RIGHT, color="#00FFFF")
        math_expr = MathTex(r"W \cdot x = h", color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Placing at 'B4' as requested by Critic
        self.place_at_grid(neuron1, "B4", scale_factor=0.8)
        self.play(FadeIn(neuron1))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Placing at 'C4' as requested by Critic
        self.place_at_grid(connection, "C4", scale_factor=1.0)
        self.play(Create(connection))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        # Placing at 'D5' as requested by Critic
        self.place_at_grid(math_expr, "D5", scale_factor=0.8)
        self.play(Write(math_expr))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        # Placing at 'E4' as requested by Critic
        self.place_at_grid(neuron2, "E4", scale_factor=0.8)
        self.play(FadeIn(neuron2))
        self.wait(2)
