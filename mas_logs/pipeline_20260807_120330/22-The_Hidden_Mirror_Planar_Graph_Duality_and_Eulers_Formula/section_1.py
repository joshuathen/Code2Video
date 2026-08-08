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
        # Data from storyboard
        title_text = "Prerequisite: Defining the Planar World"
        lecture_lines = [
            "A planar graph is drawn without crossing edges.",
            "It consists of vertices, edges, and faces.",
            "Every graph has one infinite outer face."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        ORANGE = "#FF8C00"
        YELLOW = "#FFFF00"
        WHITE_CLR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Define initial graph coordinates with crossing edges
        # V1: B2, V2: D4, V3: B4, V4: D2
        v1_pos = self.grid["B2"]
        v2_pos = self.grid["D4"]
        v3_pos = self.grid["B4"]
        v4_pos = self.grid["D2"]

        v1 = Dot(v1_pos, color=ORANGE)
        v2 = Dot(v2_pos, color=ORANGE)
        v3 = Dot(v3_pos, color=ORANGE)
        v4 = Dot(v4_pos, color=ORANGE)
        
        e1 = Line(v1_pos, v2_pos, color=ORANGE)
        e2 = Line(v2_pos, v3_pos, color=ORANGE)
        e3 = Line(v3_pos, v4_pos, color=ORANGE)
        e4 = Line(v4_pos, v1_pos, color=ORANGE)

        # Updaters for edges to follow vertices during animation
        e1.add_updater(lambda m: m.set_points_as_corners([v1.get_center(), v2.get_center()]))
        e2.add_updater(lambda m: m.set_points_as_corners([v2.get_center(), v3.get_center()]))
        e3.add_updater(lambda m: m.set_points_as_corners([v3.get_center(), v4.get_center()]))
        e4.add_updater(lambda m: m.set_points_as_corners([v4.get_center(), v1.get_center()]))

        graph = VGroup(e1, e2, e3, e4, v1, v2, v3, v4)
        self.play(Create(graph))
        self.wait(1)

        # Move vertex V2 to position A3 to eliminate crossing
        new_v2_pos = self.grid["A3"]
        self.play(v2.animate.move_to(new_v2_pos), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight from line 1 to line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Flash vertices sequentially
        for v in [v1, v2, v3, v4]:
            self.play(Indicate(v, color=WHITE_CLR, scale_factor=1.5), run_time=0.5)
        
        # Flash edges sequentially
        for e in [e1, e2, e3, e4]:
            self.play(Indicate(e, color=WHITE_CLR), run_time=0.5)

        # Represent the internal face as a polygon and flash it
        face = Polygon(
            v1.get_center(), v2.get_center(), v3.get_center(), v4.get_center(),
            fill_color=ORANGE, fill_opacity=0.4, stroke_width=0
        )
        self.play(FadeIn(face))
        self.play(Indicate(face, color=ORANGE), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight from line 2 to line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Infinite outer face pulse: Use a background rectangle spanning the right-side grid area
        # Fix: Adjusted area to prevent obstruction of lecture notes (Issue 21)
        outer_face_bg = Rectangle(
            width=6.0, height=6.0, fill_color=ORANGE, fill_opacity=0.1, stroke_width=0
        )
        self.place_in_area(outer_face_bg, 'A4', 'F6', scale_factor=1.0)
        outer_face_bg.set_z_index(-2)  # Place behind the graph and internal face
        
        self.play(FadeIn(outer_face_bg))
        # Pulse animation
        self.play(outer_face_bg.animate.set_fill(opacity=0.3), run_time=1)
        self.play(outer_face_bg.animate.set_fill(opacity=0.1), run_time=1)
        self.wait(2)
        
        # Finalize by returning highlight to normal
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
