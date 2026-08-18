from manim import *
import numpy as np

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
        # Setup layout
        title_text = "Establishing the 'Obvious' Pattern"
        lecture_lines = [
            "With one point, we have just one region.",
            "Two points and one line create two regions.",
            "Three points create four distinct regions.",
            "Four points create exactly eight regions.",
            "The sequence 1, 2, 4, 8 looks familiar."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_POINT = "#FFFF00"  # Yellow
        COLOR_CHORD = "#00FFFF"  # Cyan
        COLOR_PRED = "#FF00FF"   # Magenta

        # Asset paths
        PATH_POINT = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg"
        PATH_CHORD = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg"

        # Helpers for asset objects
        def get_point_asset(pos):
            # Scale slightly larger than a dot for visibility
            return SVGMobject(PATH_POINT).set_color(COLOR_POINT).scale(0.12).move_to(pos)

        def get_chord_asset(p1, p2):
            line = SVGMobject(PATH_CHORD).set_color(COLOR_CHORD)
            # Adjust width and orientation
            curr_width = line.width if line.width > 0 else 1.0
            dist = np.linalg.norm(p2 - p1)
            line.scale(dist / curr_width)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            line.rotate(angle)
            line.move_to((p1 + p2) / 2)
            return line

        # Visual Elements: Circle
        # Resolve Issue 24: Move circle to B1-F6 area and scale to 0.9
        circle = Circle(radius=2.0, color=WHITE)
        self.place_in_area(circle, 'B1', 'F6', scale_factor=0.9)
        center = circle.get_center()
        radius = circle.width / 2

        # Points setup (angles to ensure no triple intersections)
        angles = [90, 210, 330, 0]
        points = [center + np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), 0]) * radius for a in angles]

        self.play(Create(circle))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_POINT))
        p1 = get_point_asset(points[0])
        label_1 = Text("1", font_size=24, color=WHITE).move_to(center)
        self.play(FadeIn(p1), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_CHORD)
        )
        p2 = get_point_asset(points[1])
        chord_1_2 = get_chord_asset(points[0], points[1])
        
        # New labels for 2 regions
        l2_1 = Text("1", font_size=24).move_to(center + LEFT * 0.5 + UP * 0.4)
        l2_2 = Text("2", font_size=24).move_to(center + RIGHT * 0.5 + DOWN * 0.4)
        
        self.play(
            FadeIn(p2),
            Create(chord_1_2),
            ReplacementTransform(label_1, VGroup(l2_1, l2_2))
        )
        current_labels = VGroup(l2_1, l2_2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_POINT)
        )
        p3 = get_point_asset(points[2])
        chord_1_3 = get_chord_asset(points[0], points[2])
        chord_2_3 = get_chord_asset(points[1], points[2])
        
        # Labels for 4 regions
        l3_1 = Text("1", font_size=20).move_to(center + UP * 0.8)
        l3_2 = Text("2", font_size=20).move_to(center + LEFT * 0.8)
        l3_3 = Text("3", font_size=20).move_to(center + RIGHT * 0.8)
        l3_4 = Text("4", font_size=20).move_to(center + DOWN * 0.4)
        
        self.play(
            FadeIn(p3),
            Create(chord_1_3),
            Create(chord_2_3),
            ReplacementTransform(current_labels, VGroup(l3_1, l3_2, l3_3, l3_4))
        )
        current_labels = VGroup(l3_1, l3_2, l3_3, l3_4)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_CHORD)
        )
        p4 = get_point_asset(points[3])
        chord_1_4 = get_chord_asset(points[0], points[3])
        chord_2_4 = get_chord_asset(points[1], points[3])
        chord_3_4 = get_chord_asset(points[2], points[3])
        
        # Labels for 8 regions
        l4_pos = [
            center + UP * 1.3,
            center + LEFT * 1.1 + UP * 0.6,
            center + RIGHT * 1.1 + UP * 0.2,
            center + LEFT * 0.5 + DOWN * 0.5,
            center + RIGHT * 0.5 + DOWN * 0.8,
            center + DOWN * 1.3,
            center + UP * 0.2 + LEFT * 0.3,
            center + DOWN * 0.2 + RIGHT * 0.1
        ]
        l4_labels = VGroup(*[Text(str(i+1), font_size=18) for i in range(8)])
        for i, lbl in enumerate(l4_labels):
            lbl.move_to(l4_pos[i])
            
        self.play(
            FadeIn(p4),
            Create(chord_1_4),
            Create(chord_2_4),
            Create(chord_3_4),
            ReplacementTransform(current_labels, l4_labels)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_PRED)
        )
        
        # Resolve Issue 23: Move sequence to A2-A5 area
        seq = Text("1, 2, 4, 8, ", font_size=32)
        self.place_in_area(seq, 'A2', 'A5', scale_factor=1.0)
        pred = Text("16?", font_size=36, color=COLOR_PRED)
        pred.next_to(seq, RIGHT)
        
        self.play(Write(seq))
        self.play(FadeIn(pred))
        # Pulse animation
        self.play(pred.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.play(pred.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        
        self.wait(2)
