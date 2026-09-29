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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visual Synthesis: The Sequential Path", [
            "Watch the vector move through two transformations.",
            "First it changes under A, then B.",
            "The matrix BA performs both at once."
        ])
        
        # Elements
        v = Vector([1, 1], color=YELLOW)
        v_label = MathTex(r"\\vec{v}", color=YELLOW, font_size=24)
        vec_group = VGroup(v, v_label)
        
        label_A = MathTex("A", color=WHITE, font_size=36)
        label_B = MathTex("B", color=WHITE, font_size=36)

        # Position elements based on critique
        self.place_in_area(vec_group, 'B3', 'E5', scale_factor=1.2)
        self.place_at_grid(label_A, 'B3', scale_factor=0.6)
        self.place_at_grid(label_B, 'B5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(vec_group), FadeIn(label_A), FadeIn(label_B))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        
        v_prime = Vector([2, 0.5], color=BLUE)
        v_double_prime = Vector([0.5, 2], color=RED)
        
        self.play(Transform(v.copy(), v_prime), run_time=1.5)
        self.play(Transform(v.copy(), v_double_prime), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        
        final_v = Vector([0.5, 2], color=GREEN)
        self.play(ReplacementTransform(v, final_v), run_time=2)
