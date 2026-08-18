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
        lecture_lines = ["Recall the standard power rule.", "Consider a rectangle with sides f(x) and g(x).", "Visualize the rectangle area as f(x) times g(x)."]
        self.setup_layout("Prerequisites & Intuition", lecture_lines)
        
        # Elements
        f_x = MathTex("f(x)", color="#3498DB")
        g_x = MathTex("g(x)", color="#3498DB")
        d_dx = MathTex(r"\frac{d}{dx}", color="#2ECC71")
        mult = MathTex(r"\times", color="#F1C40F")
        area_label = MathTex("A = f(x)g(x)", color="#2ECC71")
        rectangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rectangle.svg", color="#3498DB")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#2ECC71")
        # Fixing operator position (Addressing Issue 22/37)
        self.place_at_grid(d_dx, 'B2', scale_factor=1.0)
        self.play(Write(d_dx))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        # Integration of Asset (Addressing Issue 16)
        # Group math elements for layout (Addressing Issue 23/38)
        math_group = VGroup(f_x, mult, g_x).arrange(RIGHT)
        self.place_in_area(math_group, 'C2', 'D5', scale_factor=0.9)
        self.place_at_grid(rectangle, 'E4', scale_factor=0.5)
        self.play(FadeIn(rectangle), FadeIn(math_group))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#2ECC71")
        self.place_at_grid(area_label, 'F3', scale_factor=0.8)
        self.play(FadeIn(area_label))
