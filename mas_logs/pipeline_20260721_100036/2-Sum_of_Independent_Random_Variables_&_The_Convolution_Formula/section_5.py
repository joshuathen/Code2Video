from manim import *
import numpy as np

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
        # Setup Title and Lecture Lines
        title = "Key Properties & The Central Limit Theorem Hint"
        lines = [
            "Convolution generally smooths out the resulting distribution.",
            "The sum of two uniform variables becomes triangular.",
            "Convolving again makes the distribution even smoother.",
            "Repeated convolution leads towards a bell-shaped curve.",
            "This illustrates the famous Central Limit Theorem."
        ]
        self.setup_layout(title, lines)
        
        # --- Persistent Objects ---
        # Right side axes for plotting distributions
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False, "font_size": 18, "color": GREY},
            x_length=5,
            y_length=3.5
        )
        self.place_in_area(axes, "B1", "E6")
        self.add(axes)

        # 1. Square Wave (Uniform on [-0.5, 0.5])
        square_curve = axes.plot(
            lambda x: 1.0 if abs(x) <= 0.5 else 0.0,
            discontinuities=[-0.5, 0.5],
            dt=0.01,
            color=WHITE
        )

        # 2. Triangle Wave (n=2, sum of two uniforms on [-1, 1])
        triangle_curve = axes.plot(
            lambda x: max(0, 1 - abs(x)),
            color="#ADD8E6",
            use_smoothing=False
        )

        # 3. Irwin-Hall (n=3, centered on [-1.5, 1.5])
        def n3_pdf(x):
            xp = x + 1.5
            if 0 <= xp < 1:
                return 0.5 * xp**2
            elif 1 <= xp < 2:
                return 0.5 * (-2 * (xp-1)**2 + 2*(xp-1) + 1)
            elif 2 <= xp <= 3:
                return 0.5 * (3-xp)**2
            return 0

        n3_curve = axes.plot(
            n3_pdf,
            color="#ADD8E6"
        )

        # 4. Normal Distribution (Gaussian)
        def gaussian_pdf(x):
            # Variance for sum of n uniforms U(-0.5, 0.5) is n/12. For n=4, sigma = sqrt(4/12) approx 0.57.
            # We use sigma=0.7 for better visualization on axes.
            sigma = 0.7 
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)

        gaussian_curve = axes.plot(
            gaussian_pdf,
            color="#FFFF00"
        )
        
        # Resolved Issue 50 (Fixing 41): Moved to B4 and scaled to 0.6
        normal_label = Text("Normal Distribution", font_size=22, color="#FFFF00")
        self.place_at_grid(normal_label, "B4", scale_factor=0.6)
        
        # Resolved Issue 50 (Fixing 42): Using place_in_area and scale 0.7
        clt_label = Text("Central Limit Theorem", font_size=28, color="#FF69B4")
        self.place_in_area(clt_label, "F2", "F5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Convolution generally smooths out the resulting distribution.
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(square_curve), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # The sum of two uniform variables becomes triangular.
        self.lecture[1].set_color("#ADD8E6")
        self.play(Transform(square_curve, triangle_curve), run_time=2)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Convolving again makes the distribution even smoother.
        self.lecture[2].set_color("#ADD8E6")
        self.play(Transform(square_curve, n3_curve), run_time=2)
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Repeated convolution leads towards a bell-shaped curve.
        self.lecture[3].set_color("#FFFF00")
        self.play(
            Transform(square_curve, gaussian_curve),
            Write(normal_label),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # This illustrates the famous Central Limit Theorem.
        self.lecture[4].set_color("#FF69B4")
        self.play(
            FadeIn(clt_label, shift=UP),
            Indicate(clt_label, color="#FF69B4"),
            run_time=2
        )
        self.wait(3)
