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
        lecture_lines = ["Cones are surfaces formed by rotation.", "Slicing cones creates conic sections.", "Circles, ellipses, parabolas, hyperbolas emerge."]
        self.setup_layout("Introduction: The Slicing Problem", lecture_lines)
        
        # Elements
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg", color=WHITE)
        label_cone = Text("Cone", font_size=24, color=WHITE)
        plane = Polygon(LEFT*1.5 + UP*0.5, RIGHT*1.5 + UP*0.5, RIGHT*1.5 + DOWN*0.5, LEFT*1.5 + DOWN*0.5, color=YELLOW, fill_opacity=0.3)
        intersection = Ellipse(width=1.0, height=0.5, color=PURPLE, stroke_width=4)
        label_inter = Text("Intersection", font_size=24, color=PURPLE)

        # Positioning
        self.place_in_area(cone, 'B3', 'E6', scale_factor=1.2)
        self.place_at_grid(label_cone, 'B3', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cone), Write(label_cone))
        self.play(Rotate(cone, angle=2*PI, axis=UP, run_time=2))
        self.lecture[0].set_color(LIGHT_GRAY) # Changing to a visible "highlight" color per instructions

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(plane))
        self.place_at_grid(intersection, 'D3', scale_factor=0.7)
        self.play(Create(intersection), Write(label_inter), self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(plane), self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(PURPLE))
        self.wait(2)
