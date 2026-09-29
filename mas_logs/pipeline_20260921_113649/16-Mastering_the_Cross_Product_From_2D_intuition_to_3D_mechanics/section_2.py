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
        lecture_lines = [
            "In 3D, cross product produces vectors.",
            "Result is perpendicular to both.",
            "Magnitude equals the parallelogram area.",
            "Right-hand rule determines direction.",
            "Visualizing rotation axes in 3D."
        ]
        self.setup_layout("Transition to 3D: The Geometry of Space", lecture_lines)
        
        # Setup 3D objects
        u = Arrow3D(start=ORIGIN, end=RIGHT+UP, color=WHITE)
        v = Arrow3D(start=ORIGIN, end=RIGHT*1.5+DOWN*0.5, color=WHITE)
        w = Arrow3D(start=ORIGIN, end=OUT*1.5, color="#00FF00")
        
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2])
        scene_group = VGroup(axes, u, v, w)
        # Position per issue #24, #39
        self.place_in_area(scene_group, "C2", "F6", scale_factor=0.5)

        # Labels
        vector_labels = VGroup(Text("u", color=WHITE), Text("v", color=WHITE), Text("w", color="#00FF00"))
        # Position per issue #25, #40
        self.place_at_grid(vector_labels, "D4", scale_factor=0.6)

        # Right Hand Rule Asset
        hand_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        rotation_axis_dot = Dot(color="#FFFF00").move_to(w.get_end())
        # Position per issue #26, #41
        self.place_at_grid(rotation_axis_dot, "C3", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.add(axes, u, v)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(Create(w))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Parallelogram visual
        para = Polygon(ORIGIN, u.get_end(), u.get_end()+v.get_end(), v.get_end(), color="#FF00FF", fill_opacity=0.3)
        self.play(Create(para))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(FadeIn(hand_icon.move_to(rotation_axis_dot.get_center())))
        self.play(Indicate(rotation_axis_dot))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.play(Rotate(scene_group, angle=PI/4, axis=OUT))
