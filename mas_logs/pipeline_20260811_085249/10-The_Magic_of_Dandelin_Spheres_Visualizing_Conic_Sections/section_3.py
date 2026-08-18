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
        self.setup_layout("Connecting the Geometry to the Conic", [
            "Points on the conic relate to foci.", 
            "Tangency points define key geometric properties.", 
            "These spheres prove the conic definition."
        ])
        
        # Create shapes (Using SVGMobjects as assets)
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        point_p = Dot(color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        cone_vis = VGroup(cone, plane).set_color(BLUE)
        self.place_at_grid(cone_vis, "C4", scale_factor=0.6)
        self.play(Create(cone_vis))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        t1 = Dot(color=YELLOW)
        t2 = Dot(color=YELLOW)
        label1 = Text("F1", font_size=20, color=YELLOW)
        label2 = Text("F2", font_size=20, color=YELLOW)
        
        self.place_at_grid(t1, "B2", scale_factor=1)
        self.place_at_grid(t2, "D4", scale_factor=1)
        self.place_at_grid(label1, "B3", scale_factor=0.8)
        self.place_at_grid(label2, "D4", scale_factor=0.8)
        
        self.play(FadeIn(t1), FadeIn(t2), Write(label1), Write(label2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_at_grid(sphere, "C5", scale_factor=0.5)
        self.place_at_grid(point_p, "C5", scale_factor=0.5)
        line1 = Line(t1.get_center(), point_p.get_center(), color=WHITE)
        line2 = Line(t2.get_center(), point_p.get_center(), color=WHITE)
        
        self.play(FadeIn(sphere), Create(line1), Create(line2))
        self.wait(2)
