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
        lecture_lines = [
            "Pi defines rotation and circular motion.",
            "Constrained systems inherently produce Pi.",
            "Our collisions are simply rotating in space.",
            "Data from collisions forms a perfect circle.",
            "Pi is fundamental to all oscillatory motion."
        ]
        self.setup_layout("Conclusion: Why Pi Appears Everywhere", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display the relation N = 10^k * Pi.
        eq1 = MathTex(r"N", r"=", r"10^k", r"\pi")
        eq1.set_color_by_tex(r"N", YELLOW)
        self.place_at_grid(eq1, "B2", scale_factor=1.2)
        self.play(FadeIn(eq1))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        # Show the scaling factor sqrt(m1/m2) clearly in green #00FF00.
        eq2 = MathTex(r"\sqrt{\frac{m_1}{m_2}}")
        eq2.set_color("#00FF00")
        self.place_at_grid(eq2, "B5", scale_factor=1.0)
        self.play(Write(eq2))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Data from collisions forms a perfect circle.
        circle = Circle(radius=0.5, color=RED)
        self.place_at_grid(circle, "D3", scale_factor=0.8)
        self.play(Create(circle))
        self.play(self.lecture[3].animate.set_color(RED))

        # === Animation for Lecture Line 5 ===
        # Pi is fundamental to all oscillatory motion.
        final_pi = Tex(r"$\pi$", font_size=72, color=WHITE)
        self.place_at_grid(final_pi, "E5", scale_factor=1.2)
        self.play(FadeIn(final_pi))
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
