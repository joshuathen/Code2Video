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
        self.setup_layout("The Geometry of the Shadow", [
            "Parallel light rays hit a sphere perfectly.",
            "The shadow is a circle with radius r.",
            "Its area is pi times r squared."
        ])
        
        # Setup assets
        # Using SVGMobject for icon assets as per instructions
        light = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        plane = NumberPlane().set_fill(opacity=0.5).scale(1.5)
        shadow = Circle(radius=0.5, color="#808080").set_fill(opacity=0.6)
        
        # Initial positions
        self.place_at_grid(sphere, 'B2', scale_factor=0.6)
        self.place_at_grid(light, 'A2', scale_factor=0.4)
        self.place_at_grid(plane, 'E2', scale_factor=0.7)
        self.place_at_grid(shadow, 'E4', scale_factor=0.7)
        
        # Hide objects initially
        sphere.set_opacity(0)
        light.set_opacity(0)
        plane.set_opacity(0)
        shadow.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Parallel light rays hit a sphere perfectly.
        self.play(self.lecture[0].animate.set_color("#3399FF"), 
                  FadeIn(sphere), FadeIn(light))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The shadow is a circle with radius r.
        self.play(self.lecture[1].animate.set_color("#808080"), 
                  FadeIn(plane), FadeIn(shadow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Its area is pi times r squared.
        area_text = MathTex(r"A = \pi r^2", color="#FFD700").scale(1.0)
        self.place_at_grid(area_text, 'E5', scale_factor=0.8)
        self.play(self.lecture[2].animate.set_color("#FFD700"), Write(area_text))
        self.wait(2)
