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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining Higher Order Derivatives", [
            "Second derivative is the derivative of the derivative.",
            "It measures concavity and acceleration.",
            "Positive second derivative means concave up.",
            "Negative second derivative means concave down.",
            "It tracks how velocity changes over time."
        ])
        
        # Assets
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # === Animation for Lecture Line 1 ===
        # Second derivative is the derivative of the derivative.
        self.lecture[0].set_color("#FFFFFF")
        curve = FunctionGraph(lambda x: 0.1 * x**3, x_range=[-2, 2], color=WHITE)
        self.place_in_area(curve, 'B2', 'C4')
        self.place_at_grid(car_icon.copy(), 'B2', scale_factor=0.2)
        self.play(Create(curve), FadeIn(car_icon))

        # === Animation for Lecture Line 2 ===
        # It measures concavity and acceleration.
        self.lecture[1].set_color("#00FF00")
        f_double_prime = MathTex(r"f''(x)", color="#00FFFF")
        self.place_in_area(f_double_prime, 'C3', 'D4', scale_factor=0.9)
        accel_label = Text("a = f''(x)", color="#00FF00")
        self.place_at_grid(accel_label, 'D5', scale_factor=0.6)
        self.play(Write(f_double_prime), Write(accel_label))

        # === Animation for Lecture Line 3 ===
        # Positive second derivative means concave up.
        self.lecture[2].set_color("#FFCC00")
        concavity_label = Text("Concavity", color="#FFCC00")
        self.place_at_grid(concavity_label, 'D3', scale_factor=0.7)
        self.play(FadeIn(concavity_label))

        # === Animation for Lecture Line 4 ===
        # Negative second derivative means concave down.
        self.lecture[3].set_color("#FFCC00")
        self.play(concavity_label.animate.set_color("#FF0000"))

        # === Animation for Lecture Line 5 ===
        # It tracks how velocity changes over time.
        self.lecture[4].set_color("#00FFFF")
        self.place_at_grid(car_icon.copy(), 'E5', scale_factor=0.3)
        self.play(FadeIn(car_icon))
