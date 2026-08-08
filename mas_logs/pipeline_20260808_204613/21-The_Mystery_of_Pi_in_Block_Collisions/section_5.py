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
        self.setup_layout("Conclusion and Intuition", [
            "Discrete collisions map to digits of Pi.",
            "Laws of physics constrain the system.",
            "Simple interactions compute complex mathematical truths."
        ])
        
        # --- Visual Setup ---
        # Gear representation
        gear = VGroup(
            Circle(radius=0.5, color=BLUE),
            *[Line(ORIGIN, 0.6 * UP).rotate(i * 36 * DEGREES, about_point=ORIGIN) for i in range(10)]
        ).set_color(BLUE)
        
        # Placeholder asset icons
        # Note: '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' doesn't exist, will use standard mobjects as a fallback if necessary 
        # or just acknowledge the instruction. Asset references: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon1 = Circle(radius=0.2, color=GRAY).set_fill(GRAY, opacity=0.5)
        icon2 = Square(side_length=0.4, color=GRAY).set_fill(GRAY, opacity=0.5)
        
        # --- Animation for Lecture Line 1 ---
        # Issue 40: Place gear at 'B5' (scale 0.8)
        self.place_at_grid(gear, 'B5', scale_factor=0.8)
        self.play(FadeIn(gear))
        self.place_at_grid(icon1, 'B6', scale_factor=0.5)
        self.play(FadeIn(icon1))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # --- Animation for Lecture Line 2 ---
        self.play(gear.animate.rotate(PI, about_point=self.grid['B5']))
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # --- Animation for Lecture Line 3 ---
        # Issue 40: Place pi_text in 'D4'-'F6' (scale 0.9)
        pi_text = MathTex(r"\pi = 3.14159...", color=RED)
        self.place_in_area(pi_text, 'D4', 'F6', scale_factor=0.9)
        self.play(Write(pi_text))
        self.place_at_grid(icon2, 'F6', scale_factor=0.5)
        self.play(FadeIn(icon2))
        self.play(self.lecture[2].animate.set_color(RED))
        
        # Flash final result
        self.play(pi_text.animate.set_color("#FFFF00"))
        
        self.wait(2)
