from manim import *

# Set configuration to prevent race conditions during LaTeX cleanup
config.no_latex_cleanup = True

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
        self.setup_layout("Roots: The Inverse of Power", [
            "Roots are the inverse of power.", 
            "They help find the base growth rate.", 
            "Think of it as 'unwinding' growth."
        ])
        
        # Load asset
        plant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plant.svg")
        
        # Power equation: b^x = y
        power_eq = MathTex("b^x = y", color=WHITE)
        self.place_in_area(power_eq, 'B3', 'B5', scale_factor=1.2)
        
        # Root equation: x = \sqrt[b]{y}
        root_eq = MathTex("x = \\sqrt[b]{y}", color=WHITE)
        self.place_in_area(root_eq, 'D3', 'D5', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        # Add plant icon
        plant_a = plant.copy()
        self.place_at_grid(plant_a, 'B1', scale_factor=0.5)
        self.play(Write(power_eq), FadeIn(plant_a))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Write(root_eq))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        arrow = Arrow(start=power_eq.get_bottom(), end=root_eq.get_top(), color="#FF00FF")
        self.place_at_grid(arrow, 'E4', scale_factor=0.9)
        # Add another plant icon
        plant_b = plant.copy()
        self.place_at_grid(plant_b, 'E6', scale_factor=0.5)
        self.play(Create(arrow), FadeIn(plant_b))
        self.wait(2)
