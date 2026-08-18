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
        # Fetching lecture lines from storyboard
        lecture_lines = [
            "Two blocks and a wall create a surprising mystery.",
            "We count every single collision between blocks and walls.",
            "Total collisions magically reveal the digits of Pi."
        ]
        self.setup_layout("The Strange Phenomenon", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Visual Elements: Wall SVG, Floor, Blocks
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        self.place_in_area(wall, "A1", "F1", scale_factor=1.0)
        wall.set_color(GRAY_A)
        
        floor = Line(self.grid["F1"], self.grid["F6"] + RIGHT*0.5, color=WHITE, stroke_width=4)
        
        # Small block m
        block_m = Square(side_length=0.6, fill_opacity=1, color=BLUE)
        m_label = Text("m", font_size=20, color=BLUE).next_to(block_m, UP, buff=0.1)
        m_group = VGroup(block_m, m_label)
        self.place_at_grid(m_group, "F2")
        m_group.shift(UP * 0.3) # Sit exactly on floor (F row is y=-2.8, square side 0.6)
        
        # Large block M
        block_M = Square(side_length=1.2, fill_opacity=1, color=RED)
        M_label = Text("M", font_size=24, color=RED).next_to(block_M, UP, buff=0.1)
        M_group = VGroup(block_M, M_label)
        self.place_at_grid(M_group, "F6") # Issue 37: Start at F6
        M_group.shift(UP * 0.6) # Sit exactly on floor (F row is y=-2.8, square side 1.2)
        
        self.play(FadeIn(wall), Create(floor))
        self.play(FadeIn(m_group), FadeIn(M_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Velocity arrow for large block
        # M_group is at F6 (x=5.5). Its left edge is at 5.5 - 0.6 = 4.9.
        v_arrow = Arrow(
            start=M_group.get_left() + RIGHT*0.2, 
            end=M_group.get_left() + LEFT*0.8, 
            color="#00FF00", 
            buff=0
        )
        v_label = Text("v", font_size=20, color="#00FF00").next_to(v_arrow, UP, buff=0.1)
        
        # Collision Counter
        collision_count = Integer(0, color="#FFFF00")
        counter_label = Text("Collisions: ", font_size=24, color="#FFFF00")
        counter_group = VGroup(counter_label, collision_count).arrange(RIGHT)
        self.place_at_grid(counter_group, "A5") # Issue 37: Move to A5
        
        self.play(GrowArrow(v_arrow), FadeIn(v_label), Write(counter_group))
        
        # Initial Collisions sequence
        # Collision 1: M hits m
        # m is at x=1.5. m's right edge is 1.5 + 0.3 = 1.8.
        # M moves so its left edge (x - 0.6) hits 1.8. M x = 2.4.
        self.play(
            M_group.animate.move_to([2.4, M_group.get_y(), 0]),
            v_arrow.animate.shift(LEFT * (M_group.get_x() - 2.4)),
            v_label.animate.shift(LEFT * (M_group.get_x() - 2.4)),
            run_time=1.5
        )
        collision_count.set_value(1)
        self.play(Flash(m_group.get_right(), color=YELLOW, flash_radius=0.3))
        
        # Collision 2: m hits wall
        # Wall x = 0.5. m's left edge (x - 0.3) hits 0.5. m x = 0.8.
        self.play(
            m_group.animate.move_to([0.8, m_group.get_y(), 0]),
            run_time=0.6
        )
        collision_count.set_value(2)
        # Using SVG wall boundary for flash relative position (edge is at x=0.5, block hit at y=-2.5)
        self.play(Flash([0.5, -2.5, 0], color=YELLOW, flash_radius=0.3))
        
        # Collision 3: m hits M
        # M is at 2.4. Left edge is 1.8. m moves so its right edge (x + 0.3) hits 1.8. m x = 1.5.
        self.play(
            m_group.animate.move_to([1.5, m_group.get_y(), 0]),
            run_time=0.6
        )
        collision_count.set_value(3)
        self.play(Flash(M_group.get_left(), color=YELLOW, flash_radius=0.3))
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Mass Ratio reveal
        ratio_text = Text("Mass Ratio 1 : 10,000", font_size=24, color=WHITE)
        self.place_at_grid(ratio_text, "B5") # Issue 37: Move to B5
        self.play(FadeIn(ratio_text))
        
        # Rapid increase to 314
        # We use a ValueTracker to animate the count
        count_tracker = ValueTracker(3)
        collision_count.add_updater(lambda d: d.set_value(int(count_tracker.get_value())))
        
        self.play(
            count_tracker.animate.set_value(314),
            M_group.animate.shift(LEFT * 1.2),
            v_arrow.animate.shift(LEFT * 1.2),
            v_label.animate.shift(LEFT * 1.2),
            m_group.animate.set_opacity(0.4).move_to([0.8, m_group.get_y(), 0]), # Bouncing blur effect
            run_time=4,
            rate_func=linear
        )
        collision_count.remove_updater(collision_count.updaters[0])
        collision_count.set_value(314)
        m_group.set_opacity(1)
        
        self.play(Indicate(collision_count, scale_factor=1.5, color=YELLOW))
        self.wait(3)
