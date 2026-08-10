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
        self.setup_layout("Expanding to Parabola and Hyperbola", [
            "Change the plane angle to shift shapes.",
            "Parabolas use one sphere tangent to plane.",
            "Hyperbolas require spheres in both cones."
        ])

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        plane = Polygon(LEFT*1.5, RIGHT*1.5, RIGHT*1.5+UP*0.5, LEFT*1.5+UP*0.5, color=BLUE, fill_opacity=0.5)

        cone_and_plane = VGroup(cone, plane)
        animation_group = VGroup(cone_and_plane)

        # Stage 1
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(animation_group, 'B4', 'F6', scale_factor=0.85)
        self.play(FadeIn(cone), Create(plane))
        self.play(plane.animate.rotate(0.5, about_point=animation_group.get_center()))

        # Stage 2
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(sphere, 'E4', scale_factor=0.7)
        self.play(FadeIn(sphere))
        
        # Stage 3
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        sphere2 = sphere.copy().move_to(self.grid['B4'])
        self.play(FadeIn(sphere2), cone.animate.rotate(0.2, axis=OUT))
        
        self.wait(2)
