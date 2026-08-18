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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Determinant represents area of a parallelogram.", 
                         "Vectors spanning space define an area.", 
                         "Non-zero determinant means vectors are independent."]
        self.setup_layout("Prerequisite Review: Determinants as Area", lecture_lines)
        
        # Create axes
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": True}).scale(0.7)
        self.place_at_grid(axes, 'D2', scale_factor=0.9)
        
        v1 = Vector([2, 0], color=BLUE)
        v2 = Vector([0, 2], color=RED)
        
        # Using SVG Asset
        parallelogram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg", color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(axes), Create(v1), Create(v2))
        self.place_at_grid(parallelogram, 'D2', scale_factor=1.0)
        self.play(FadeIn(parallelogram))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        label_area = Text("Area = 4", color="#FFD700", font_size=20)
        self.place_at_grid(label_area, 'B5', scale_factor=0.7)
        self.play(Write(label_area))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        det_label = MathTex(r"det(A) \neq 0", color="#00FFFF")
        self.place_at_grid(det_label, 'C5', scale_factor=0.8)
        self.play(Write(det_label))
        
        self.wait(2)
