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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Which path is the fastest between two points?",
            "Is it the straightest or the steepest path?",
            "Let's compare a straight line and a curve.",
            "Intuitively, the straightest path seems the fastest.",
            "But physics often defies our simple intuition."
        ]
        self.setup_layout("The Great Race: Intuition vs. Reality", lecture_lines)
        
        # Set initial colors (dimmed)
        for line in self.lecture:
            line.set_color(GRAY)

        # Grid positions for points
        pos_A = self.grid['B1']
        pos_B = self.grid['E6']
        
        # Asset path
        marble_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marbles.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)
        
        point_A = Dot(pos_A, color=WHITE)
        label_A = Text("A", font_size=20, color=WHITE).next_to(point_A, UP, buff=0.2)
        point_B = Dot(pos_B, color=WHITE)
        label_B = Text("B", font_size=20, color=WHITE).next_to(point_B, DOWN, buff=0.2)
        
        self.play(
            Create(point_A), 
            Write(label_A), 
            Create(point_B), 
            Write(label_B), 
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)
        
        # Paths
        # Straight (#FFFFFF)
        path_straight = Line(pos_A, pos_B, color=WHITE)
        # Circular Arc (#FFD700)
        path_circular = ArcBetweenPoints(pos_A, pos_B, radius=6, color="#FFD700")
        # Steep Curve (#00BFFF)
        path_steep = CubicBezier(
            pos_A, 
            pos_A + 3 * DOWN, 
            pos_B + 1.5 * LEFT, 
            pos_B, 
            color="#00BFFF"
        )
        
        self.play(
            Create(path_straight), 
            Create(path_circular), 
            Create(path_steep), 
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE), run_time=0.5)
        
        # Marbles (using SVGMobject as per Issue 23)
        marble_straight = SVGMobject(marble_asset).scale(0.2).set_color(WHITE).move_to(pos_A)
        marble_circular = SVGMobject(marble_asset).scale(0.2).set_color("#FFD700").move_to(pos_A)
        marble_steep = SVGMobject(marble_asset).scale(0.2).set_color("#00BFFF").move_to(pos_A)
        
        self.add(marble_straight, marble_circular, marble_steep)
        
        # Trackers for movement (Persistent mobjects + updaters)
        track_straight = ValueTracker(0)
        track_circular = ValueTracker(0)
        track_steep = ValueTracker(0)
        
        marble_straight.add_updater(lambda m: m.move_to(path_straight.point_from_proportion(track_straight.get_value())))
        marble_circular.add_updater(lambda m: m.move_to(path_circular.point_from_proportion(track_circular.get_value())))
        marble_steep.add_updater(lambda m: m.move_to(path_steep.point_from_proportion(track_steep.get_value())))
        
        # Animate descent: Steep (2s), Circular (3s), Straight (4s)
        self.play(
            track_steep.animate.set_value(1),
            track_circular.animate.set_value(1),
            track_straight.animate.set_value(1),
            run_time=4,
            rate_func=linear
        )
        
        # Remove updaters to finalize positions
        marble_straight.clear_updaters()
        marble_circular.clear_updaters()
        marble_steep.clear_updaters()
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00BFFF"), run_time=0.5)
        
        # Highlight Blue path
        label_fastest = Text("Fastest Path", font_size=20, color="#00BFFF")
        # Issue 30: Place at D5 with scale_factor 0.8
        self.place_at_grid(label_fastest, "D5", scale_factor=0.8)
        
        self.play(
            path_steep.animate.set_stroke(width=6),
            Write(label_fastest),
            run_time=1
        )
        self.wait(2)
