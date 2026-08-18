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
        lecture_lines = [
            "Parabolas and hyperbolas also use this tangency principle.",
            "Shifting the plane morphs the resulting conic section.",
            "Tangency remains the key to all conic behavior."
        ]
        self.setup_layout("Generalization and Conclusion", lecture_lines)
        
        # Load Assets
        plane_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg").set_color(YELLOW)
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(BLUE)
        
        # Labels
        plane_label = Text("Plane", font_size=20, color=YELLOW)
        sphere_label = Text("Sphere", font_size=20, color=BLUE)
        
        # Visualization (Applied fixes per issue 26, 27, 28)
        self.place_in_area(plane_icon, "B5", "C6", scale_factor=0.7)
        self.place_in_area(sphere_icon, "D5", "E6", scale_factor=0.7)
        self.place_at_grid(plane_label, "B4", scale_factor=0.6)
        plane_label.next_to(plane_icon, LEFT, buff=0.1)
        self.place_at_grid(sphere_label, "D4", scale_factor=0.6)
        sphere_label.next_to(sphere_icon, LEFT, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Rotate(plane_icon, angle=PI/4, about_point=self.grid["B5"]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(sphere_icon.animate.shift(LEFT * 0.5), sphere_label.animate.shift(LEFT * 0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeToColor(VGroup(plane_icon, sphere_icon, plane_label, sphere_label), WHITE))
        self.wait(2)
