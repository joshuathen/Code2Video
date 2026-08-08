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
        self.setup_layout("Introduction: The Chaos of the Coffee Cup", [
            "Turbulence is a multi-scale energy cascade.", 
            "Reynolds number dictates laminar to turbulent transition.", 
            "Inertial forces dominate over viscous forces in turbulence."
        ])
        
        # Load Assets
        cup = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cup.svg")
        milk = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/milk.svg")
        coffee_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg")
        
        # Create visual elements
        self.place_in_area(cup, 'B2', 'D4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(cup), FadeIn(milk))
        
        turbulence_label = Text("Turbulence", color="#87CEEB", font_size=36)
        self.place_at_grid(turbulence_label, 'B3', scale_factor=0.7)
        self.play(FadeIn(turbulence_label))
        
        # Eddy animation
        eddy = Circle(radius=0.4, color=WHITE, fill_opacity=0.1)
        self.place_at_grid(eddy, 'E3', scale_factor=0.6)
        self.play(Rotate(eddy, angle=2*PI, run_time=2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        eddy2 = Circle(radius=0.2, color=WHITE, fill_opacity=0.1)
        self.place_at_grid(eddy2, 'E4', scale_factor=0.6)
        self.play(Create(eddy2), Rotate(eddy2, angle=-2*PI, run_time=1.5))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        smallest_eddy = coffee_icon
        self.place_at_grid(smallest_eddy, 'E5', scale_factor=0.5)
        self.play(Flash(smallest_eddy, color=RED, flash_radius=0.3))
        self.wait(1)
