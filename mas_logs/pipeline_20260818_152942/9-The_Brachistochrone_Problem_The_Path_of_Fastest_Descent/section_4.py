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
        lecture_lines = [
            "We minimize total time as a functional.",
            "Euler-Lagrange equations find the stationary point.",
            "This calculus proves the cycloid is fastest.",
            "Small path deviations increase the travel time.",
            "Nature optimizes paths to reach the minimum."
        ]
        self.setup_layout("The Calculus of Variations", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        time_integral = MathTex(r"T = \int_{A}^{B} \frac{ds}{v(y)}", color=WHITE)
        self.place_at_grid(time_integral, 'B2', scale_factor=1.2)
        self.play(Write(time_integral))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        e_l_eq = MathTex(r"\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0", color="#FFD700")
        self.place_at_grid(e_l_eq, 'D3', scale_factor=0.8)
        self.play(FadeIn(e_l_eq))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        # Represent a path variation
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 2, 1], axis_config={"include_tip": False}, x_length=3, y_length=2)
        curve = axes.plot(lambda x: 0.5 * (1 - np.cos(x*2)), color="#FF4500")
        self.place_at_grid(axes, 'B4', scale_factor=0.5)
        # Note: Manim's place_at_grid scales the object relative to its original size.
        # Since we just created it, it should work fine.
        curve.move_to(axes.get_center())
        self.play(Create(axes), Create(curve))
        self.lecture[2].set_color("#FF4500")

        # === Animation for Lecture Line 4 ===
        # Show variation
        variation = axes.plot(lambda x: 0.5 * (1 - np.cos(x*2)) + 0.2 * np.sin(x*np.pi), color="#00FFFF")
        self.play(Transform(curve, variation))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        result = Text("Stationary = Minimum Time", color="#00FF00")
        self.place_at_grid(result, 'C5', scale_factor=0.6)
        self.play(Write(result))
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
