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
        lecture_lines = ["Light speed changes based on the material.", "Denser materials slow down light significantly.", "Refractive index measures this speed change."]
        self.setup_layout("Prerequisite: The Speed of Light and Optical Density", lecture_lines)
        
        # Assets
        flashlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg").set_color("#FFFFFF")
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg").set_color("#00BFFF")
        
        light_ray_air = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#FFFFFF", stroke_width=6)
        light_ray_water = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#00BFFF", stroke_width=6)
        
        label_air = Text("Air (Fast)", font_size=20, color="#FFFFFF")
        label_water = Text("Water (Slow)", font_size=20, color="#00BFFF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(flashlight, 'B2', scale_factor=0.6)
        self.place_at_grid(light_ray_air, 'B3', scale_factor=0.8)
        self.place_at_grid(label_air, 'B2', scale_factor=0.8) # Adjusted label positioning
        label_air.next_to(flashlight, UP, buff=0.1)
        self.play(FadeIn(flashlight), Create(light_ray_air), Write(label_air))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(water, 'E2', scale_factor=0.6)
        self.place_at_grid(light_ray_water, 'E3', scale_factor=0.8)
        self.place_at_grid(label_water, 'E2', scale_factor=0.8)
        label_water.next_to(water, UP, buff=0.1)
        self.play(FadeIn(water), Create(light_ray_water), Write(label_water))
        self.lecture[1].set_color("#00BFFF")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        n_formula = MathTex(r"n = \frac{c}{v}", font_size=40)
        self.place_in_area(n_formula, 'C4', 'E4', scale_factor=0.9)
        self.play(Write(n_formula))
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
