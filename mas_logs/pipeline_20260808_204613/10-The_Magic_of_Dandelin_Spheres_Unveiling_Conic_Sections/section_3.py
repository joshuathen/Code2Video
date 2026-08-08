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
        lecture_lines = [
            "Place a sphere inside the cone.",
            "The sphere touches the cone's inner circle.",
            "The sphere also touches the slicing plane.",
            "The contact point is the conic focus.",
            "This creates the elegant Dandelin construction."
        ]
        self.setup_layout("The Dandelin Construction", lecture_lines)
        
        # Elements using assets
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_fill(BLUE, opacity=0.3)
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_fill("#87CEEB", opacity=0.7)
        plane = Polygon(LEFT*2+UP*0.5, RIGHT*2+UP*0.5, RIGHT*2+DOWN*0.5, LEFT*2+DOWN*0.5).rotate(PI/6, axis=RIGHT).set_fill(GRAY, opacity=0.5)
        cone_structure = VGroup(cone, plane)
        focus_dot = Dot(color="#FF4500")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(cone_structure, 'A4', 'C6', scale_factor=0.7)
        self.place_in_area(sphere, 'B4', 'E6', scale_factor=0.6)
        self.play(FadeIn(cone_structure), FadeIn(sphere))
        self.lecture[0].set_color("#87CEEB")

        # === Animation for Lecture Line 2 ===
        tangent_circle = Circle(radius=0.3, color="#87CEEB").move_to(sphere.get_center())
        self.play(Create(tangent_circle))
        self.lecture[1].set_color("#87CEEB")

        # === Animation for Lecture Line 3 ===
        self.play(sphere.animate.move_to(plane.get_center() + UP*0.2))
        self.lecture[2].set_color("#87CEEB")

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(focus_dot, 'D4', scale_factor=0.5)
        focus_label = Text("F", font_size=20, color="#FF4500").next_to(focus_dot, UP)
        self.play(FadeIn(focus_dot), Write(focus_label))
        self.lecture[3].set_color("#FF4500")

        # === Animation for Lecture Line 5 ===
        final_group = VGroup(cone_structure, sphere, focus_dot, focus_label)
        self.play(Rotate(final_group, angle=PI/8, axis=UP))
        self.lecture[4].set_color("#FF4500")
