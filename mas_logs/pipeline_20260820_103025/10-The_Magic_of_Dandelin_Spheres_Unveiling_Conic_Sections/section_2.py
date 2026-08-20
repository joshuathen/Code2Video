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
        self.setup_layout("Introducing the Dandelin Spheres", [
            "Dandelin spheres fit perfectly inside a hollow cone.", 
            "Each sphere touches the cone's wall perfectly.", 
            "A plane cuts both the cone and sphere."
        ])
        
        # Define objects
        # Loading SVG assets as per storyboard
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg", color=WHITE)
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FFCC00")
        plane = Polygon(LEFT*0.5 + DOWN*0.5, RIGHT*0.5 + DOWN*0.5, RIGHT*0.5 + UP*0.5, LEFT*0.5 + UP*0.5, 
                        fill_opacity=0.4, color="#FF0000")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fix for Issue 25/37: cone to C3-E5
        self.place_in_area(cone, "C3", "E5", scale_factor=0.8)
        self.play(FadeIn(cone))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Fix for Issue 26/37: sphere to D4, tangent_mark to D5
        self.place_at_grid(sphere, "D4", scale_factor=0.6)
        tangent_mark = Dot(color=WHITE, radius=0.05)
        self.place_at_grid(tangent_mark, "D5", scale_factor=0.5)
        self.play(FadeIn(sphere), FadeIn(tangent_mark))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Fix for Issue 24/37: plane to B1-E5
        self.place_in_area(plane, "B1", "E5", scale_factor=0.6)
        self.play(FadeIn(plane))
