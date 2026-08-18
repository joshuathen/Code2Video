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
            "Dot product measures how similar two words are.",
            "High dot product indicates strong contextual relevance.",
            "Softmax normalizes these scores into attention weights.",
            "Weights determine how much 'focus' to apply.",
            "Together, they map out relationships between all words."
        ]
        self.setup_layout("The Math: Dot Product and Softmax", lecture_lines)
        
        # Elements
        dot_op = MathTex(r"\mathbf{Q} \cdot \mathbf{K}^T", color="#33FF57")
        softmax_curve = FunctionGraph(lambda x: np.exp(x) / (1 + np.exp(x)), x_range=[-3, 3], color="#FF5733")
        attention_matrix = Matrix([[0.8, 0.2], [0.1, 0.9]], color="#3357FF").scale(0.5)

        # Assets
        asset_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        asset_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#33FF57"), Write(self.place_at_grid(dot_op, 'C2', scale_factor=0.9)))
        self.play(FadeIn(asset_1.next_to(dot_op, UP)))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.play(Indicate(dot_op))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF5733"), Create(self.place_in_area(softmax_curve, 'E2', 'F5', scale_factor=0.8)))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#3357FF"), FadeIn(self.place_in_area(attention_matrix, 'B4', 'C6', scale_factor=0.6)))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(FadeIn(asset_2.next_to(attention_matrix, RIGHT)))
        self.wait(2)
