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
        lecture_lines = ["Minimize time using calculus.", "Euler-Lagrange equations provide answers.", "Searching for the optimal function."]
        self.setup_layout("The Mathematical Essence: Calculus of Variations", lecture_lines)
        
        # Prepare Mobjects
        time_integral = MathTex(r"T = \int_{x_1}^{x_2} \frac{\sqrt{1+(y')^2}}{\sqrt{2gy}} dx", color=WHITE)
        euler_eq = MathTex(r"\frac{\partial f}{\partial y} - \frac{d}{dx} \left( \frac{\partial f}{\partial y'} \right) = 0", color="#FFD700")
        
        # === Animation for Lecture Line 1 ===
        # Display the functional integral for travel time (#FFFFFF)
        self.place_in_area(time_integral, 'B2', 'B5', scale_factor=0.8)
        self.play(Write(time_integral))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the Euler-Lagrange equation appearing below (#FFD700)
        self.place_in_area(euler_eq, 'D2', 'D5', scale_factor=0.75)
        self.play(Write(euler_eq))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # Highlight variables in integral (#00FFFF)
        time_integral.set_color_by_tex("y", "#00FFFF")
        time_integral.set_color_by_tex("y'", "#00FFFF")
        self.play(Indicate(time_integral))
        self.wait(1)
        
        # Animate terms cancelling (#FF4500)
        # Using a simple frame/rectangle as a "cancel" animation placeholder
        cancel_rect = Rectangle(color="#FF4500", width=1.5, height=0.5).move_to(euler_eq.get_center())
        self.play(Create(cancel_rect))
        self.play(FadeOut(cancel_rect))
        self.lecture[2].set_color("#7FFF00")
        self.wait(1)
