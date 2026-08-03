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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Prerequisite: Conservation Laws"
        lecture_lines = [
            "Two physical rules govern every single \"click.\"",
            "Conservation of energy keeps the total budget constant.",
            "Momentum transfers between blocks during each impact."
        ]
        
        # Colors
        COLOR_KE = "#00FFFF"  # Cyan
        COLOR_MOM = "#FF8000" # Orange
        COLOR_FLASH = "#FFFFFF" # White
        COLOR_INACTIVE = GRAY
        
        self.setup_layout(title, lecture_lines)
        
        # Initial state: Dim the lecture lines
        for line in self.lecture:
            line.set_color(COLOR_INACTIVE)
            
        # === Prepare Equations ===
        # KE: 1/2mv^2 + 1/2MV^2 = E
        ke_eq = MathTex(r"\frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E", color=COLOR_KE)
        # Fix Issue 23: Reduce scale factor to 1.0 for better balance
        self.place_in_area(ke_eq, 'B2', 'B5', scale_factor=1.0)
        
        # Momentum: mv + MV = P
        mom_eq = MathTex(r"mv + MV = P", color=COLOR_MOM)
        # Fix Issue 24: Move to row C for better vertical alignment and reduce scale factor to 1.0
        self.place_in_area(mom_eq, 'C2', 'C5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # "Two physical rules govern every single \"click.\""
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            run_time=1
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "Conservation of energy keeps the total budget constant."
        # Display Kinetic Energy equation in cyan
        self.play(
            self.lecture[0].animate.set_color(COLOR_INACTIVE),
            self.lecture[1].animate.set_color(COLOR_KE),
            Write(ke_eq),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Momentum transfers between blocks during each impact."
        # Display Momentum equation in orange
        self.play(
            self.lecture[1].animate.set_color(COLOR_INACTIVE),
            self.lecture[2].animate.set_color(COLOR_MOM),
            Write(mom_eq),
            run_time=1.5
        )
        self.wait(1)
        
        # Flash both equations in white during a "simulated collision"
        flash_ke = ke_eq.copy().set_color(COLOR_FLASH)
        flash_mom = mom_eq.copy().set_color(COLOR_FLASH)
        
        # Simulated collision effect: quick flash
        self.play(
            FadeIn(flash_ke),
            FadeIn(flash_mom),
            run_time=0.2,
            rate_func=there_and_back
        )
        # Clean up flash copies
        self.remove(flash_ke, flash_mom)
        
        self.wait(2)
        
        # Final highlight
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            run_time=1
        )
        self.wait(3)
