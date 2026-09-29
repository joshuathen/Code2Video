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
        self.setup_layout("Color Dependence: Dispersion", [
            "Light color relates to its frequency.",
            "Oscillation amplitude depends on light frequency.",
            "Resonance occurs near the natural frequency.",
            "Amplitude peaks significantly at resonance.",
            "This defines the dispersion curve."
        ])
        
        # Prism using asset
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color=WHITE)
        self.place_at_grid(prism, 'C2', scale_factor=0.9)
        
        incident_ray = Line(LEFT*2, ORIGIN, color=WHITE)
        ray_path = VGroup(incident_ray)
        
        # Animations
        self.play(FadeIn(prism), Create(incident_ray))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Splitting
        red_ray = Line(prism.get_right(), prism.get_right()+RIGHT*2+UP*0.5, color=RED)
        green_ray = Line(prism.get_right(), prism.get_right()+RIGHT*2, color=GREEN)
        blue_ray = Line(prism.get_right(), prism.get_right()+RIGHT*2+DOWN*0.5, color=BLUE)
        
        self.play(Create(red_ray), Create(green_ray), Create(blue_ray))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        label_r = Text("Red", color=RED, font_size=20)
        label_g = Text("Green", color=GREEN, font_size=20)
        label_b = Text("Blue", color=BLUE, font_size=20)
        self.place_at_grid(label_r, 'B4', scale_factor=0.8)
        self.place_at_grid(label_g, 'C4', scale_factor=0.8)
        self.place_at_grid(label_b, 'D4', scale_factor=0.8)
        self.play(FadeIn(label_r), FadeIn(label_g), FadeIn(label_b))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)
