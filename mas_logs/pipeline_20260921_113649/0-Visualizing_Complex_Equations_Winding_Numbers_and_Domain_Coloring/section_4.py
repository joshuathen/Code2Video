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
        self.setup_layout("Numerical Algorithms: Finding the Roots", [
            "Argument principle detects roots numerically.", 
            "Integrate winding numbers over small regions.", 
            "Computers locate roots without algebraic solving."
        ])
        
        # Asset path
        computer_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg"
        
        # === Animation for Lecture Line 1 ===
        # Show Newton iteration formula: z_new = z - f(z)/f'(z) on the computer icon.
        computer_icon = SVGMobject(computer_icon_path, color=WHITE)
        self.place_at_grid(computer_icon, "A5", scale_factor=0.5)
        
        formula = MathTex(r"z_{n+1} = z_n - \frac{f(z_n)}{f'(z_n)}", font_size=40, color=YELLOW)
        # Applying correction from Issue 29/44
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        
        self.play(FadeIn(computer_icon), Write(formula))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Plot initial guess point, draw line to next iteration, animate path using the computer icon simulation.
        # Applying correction from Issue 30/45
        dot = Dot(color=BLUE)
        self.place_at_grid(dot, 'F2', scale_factor=0.8)
        self.play(FadeIn(dot))
        
        # Applying correction from Issue 31/46
        target = Dot(color=RED)
        self.place_at_grid(target, 'F5', scale_factor=1.0)
        
        path = Line(dot.get_center(), target.get_center(), color=BLUE)
        self.play(Create(path), computer_icon.animate.shift(LEFT * 0.5))
        self.play(dot.animate.move_to(target.get_center()))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Highlight the final root position clearly.
        root_marker = Cross(color=RED)
        # Applying correction from Issue 31/46
        self.place_at_grid(root_marker, 'F6', scale_factor=0.5)
        self.play(GrowFromCenter(root_marker))
        self.lecture[2].set_color(RED)
        
        self.wait(2)
