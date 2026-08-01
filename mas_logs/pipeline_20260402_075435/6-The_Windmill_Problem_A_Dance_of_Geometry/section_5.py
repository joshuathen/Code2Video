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

class Section5Scene(TeachingScene):
    def construct(self):
        # Define lecture lines and colors
        l_lines = [
            "Let’s rotate the laser by 180 degrees.",
            "The progress bar tracks our total rotation.",
            "Each star hit becomes a new pivot point.",
            "After a half-turn, the line is parallel again.",
            "This journey visits every single star in the field."
        ]
        color_l1 = PINK
        color_l2 = BLUE_B
        color_l3 = "#FFD700"  # Gold
        color_l4 = GREEN_B
        color_l5 = WHITE

        self.setup_layout("The 180-Degree Flip", l_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_l1)
        
        # Create 7 stars scattered within A2-D5 (Issue 39)
        stars = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(7)])
        # Manually distribute them inside the specified area before calling place_in_area
        stars[0].move_to([-0.5, 0.8, 0])
        stars[1].move_to([0.8, 1.2, 0])
        stars[2].move_to([-1.2, -0.5, 0])
        stars[3].move_to([0.2, 0.1, 0])
        stars[4].move_to([1.5, -0.8, 0])
        stars[5].move_to([-0.3, -1.2, 0])
        stars[6].move_to([1.1, 0.4, 0])
        
        self.place_in_area(stars, 'A2', 'D5', scale_factor=0.8)
        self.add(stars)

        # Laser Asset Integration (Issue 26)
        laser_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/laser.svg").scale(0.2)
        laser_beam = Line(LEFT*2.2, RIGHT*2.2, color=color_l1, stroke_width=4)
        laser_group = VGroup(laser_beam, laser_icon)
        
        # Initial placement using area constraint (Issue 37)
        self.place_in_area(laser_group, 'A2', 'E6', scale_factor=0.9)
        
        angle_tracker = ValueTracker(0)
        current_pivot_idx = [0] # List for mutable index in updaters
        
        def update_laser(m):
            pivot_pos = stars[current_pivot_idx[0]].get_center()
            m[0].set_angle(-angle_tracker.get_value() * DEGREES)
            m[0].move_to(pivot_pos)
            m[1].move_to(pivot_pos)

        laser_group.add_updater(update_laser)
        self.add(laser_group)
        self.play(Create(laser_beam), FadeIn(laser_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_l2)
        
        # Progress bar (semi-circle) at F4-F6 (Issue 38)
        progress_bg = Arc(radius=0.7, start_angle=PI, angle=-PI, color=GREY_D)
        self.place_in_area(progress_bg, 'F4', 'F6', scale_factor=0.8)
        
        # Dynamic filling arc for progress
        progress_fill = Arc(radius=0.7, start_angle=PI, angle=0, color=color_l2)
        progress_fill.add_updater(
            lambda m: m.become(
                Arc(radius=0.7, start_angle=PI, angle=-angle_tracker.get_value() * DEGREES, color=color_l2)
                .move_to(progress_bg.get_center())
            )
        )
        
        progress_label = Text("0°", font_size=18, color=color_l2)
        progress_label.next_to(progress_bg, DOWN, buff=0.1)
        progress_label.add_updater(lambda m: m.become(
            Text(f"{int(angle_tracker.get_value())}°", font_size=18, color=color_l2)
            .next_to(progress_bg, DOWN, buff=0.1)
        ))

        self.add(progress_bg, progress_fill, progress_label)
        self.play(FadeIn(progress_bg), FadeIn(progress_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_l3)
        
        # Pivot swap sequence
        # We simulate the 180 degree rotation by defining angles where pivots change
        segments = [
            {"target": 30, "new_pivot": 1},
            {"target": 65, "new_pivot": 6},
            {"target": 95, "new_pivot": 3},
            {"target": 130, "new_pivot": 4},
            {"target": 160, "new_pivot": 2},
            {"target": 180, "new_pivot": 5},
        ]
        
        visited_markers = VGroup()
        # Mark first pivot
        first_marker = Circle(radius=0.15, color=color_l3, stroke_width=4).move_to(stars[0])
        self.add(first_marker)
        visited_markers.add(first_marker)

        for seg in segments:
            # Rotate
            self.play(
                angle_tracker.animate.set_value(seg["target"]),
                run_time=1.2,
                rate_func=linear
            )
            # Swap pivot and mark
            current_pivot_idx[0] = seg["new_pivot"]
            marker = Circle(radius=0.15, color=color_l3, stroke_width=4).move_to(stars[seg["new_pivot"]])
            self.add(marker)
            visited_markers.add(marker)

        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(color_l4)
        
        # Flash the parallel state at 180 degrees
        flash_line = Line(LEFT*3, RIGHT*3, color=color_l4, stroke_width=6)
        flash_line.move_to(stars[5].get_center()).set_angle(-180*DEGREES)
        self.play(Create(flash_line), run_time=0.5)
        self.play(FadeOut(flash_line), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(color_l5)
        
        # Show all 7 points are gold-marked
        self.play(
            visited_markers.animate.scale(1.3).set_stroke(width=6),
            stars.animate.set_color(color_l3),
            run_time=1
        )
        self.play(visited_markers.animate.scale(1/1.3), run_time=1)
        
        self.wait(3)
