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
        lecture_lines = ["Refraction is light changing direction.", "This happens due to speed changes.", "Snell's Law calculates these angles."]
        self.setup_layout("Defining Refraction & Snell's Law", lecture_lines)
        
        # Load Assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        
        # Define objects
        interface = Line(LEFT*1.5, RIGHT*1.5, color=GRAY)
        normal = DashedLine(UP*1.5, DOWN*1.5, color="#00FFFF")
        incident_ray = Line(UP*1.5+LEFT*1.5, ORIGIN, color=YELLOW)
        refracted_ray = Line(ORIGIN, DOWN*1.5+RIGHT*1, color=YELLOW)
        normal_label = Text("Normal", font_size=18, color="#00FFFF")
        
        refraction_diagram = VGroup(glass, interface, normal, incident_ray, refracted_ray, laser)
        
        # Grid placement (Cols 4-6) per B002
        self.place_in_area(glass, 'C4', 'E6', scale_factor=0.6)
        self.place_in_area(interface, 'C4', 'C6', scale_factor=0.6)
        self.place_in_area(normal, 'B5', 'E5', scale_factor=0.6)
        self.place_at_grid(normal_label, 'B5', scale_factor=0.8)
        
        # Equation
        snells_law = MathTex(r"n_1 \cdot \sin(\theta_1) = n_2 \cdot \sin(\theta_2)", color=WHITE)
        
        # Animation sequence
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Create(glass), Create(interface), Create(normal), FadeIn(normal_label))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(Create(laser), Create(incident_ray), Create(refracted_ray))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.place_in_area(snells_law, 'B4', 'C6', scale_factor=0.8) # Adjusted for issue 39
        self.play(Write(snells_law))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
