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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Application", [
            "Determinant is a core diagnostic tool.",
            "Non-zero means a unique solution.",
            "Zero means the system is singular."
        ])

        # Objects
        square = Square(side_length=2, color=BLUE)
        det_label = MathTex(r"\\det(A) = k", font_size=36)
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # === Animation for Lecture Line 1: Determinant is a core diagnostic tool ===
        self.lecture[0].set_color(YELLOW)
        # Apply critic fix: B3 and B4-B6
        self.place_at_grid(square, 'B3', scale_factor=0.7)
        self.place_in_area(det_label, 'B4', 'B6', scale_factor=0.9)
        self.place_at_grid(ruler_icon, 'E2', scale_factor=0.5)
        self.play(Create(square), Write(det_label), FadeIn(ruler_icon))
        self.wait(1)

        # === Animation for Lecture Line 2: Non-zero means a unique solution ===
        self.lecture[1].set_color(GREEN)
        self.play(square.animate.scale(1.5), run_time=1.5)
        self.play(Indicate(det_label))
        self.wait(1)

        # === Animation for Lecture Line 3: Zero means the system is singular ===
        self.lecture[2].set_color(RED)
        # Apply critic fix: D3
        zero_label = Text("Singular! Determinant = 0", color=RED, font_size=24)
        self.place_at_grid(zero_label, 'D3', scale_factor=0.8)
        self.play(square.animate.scale(0.01), FadeOut(square), run_time=1.5)
        self.play(Write(zero_label))
        self.place_at_grid(ruler_icon, 'E5', scale_factor=0.5) # B013: Summary scene distribution
        self.wait(2)
