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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Experiment: The Power of Repetition",
            [
                "We repeat this sampling process thousands of times.",
                "Each group's average is recorded as a single data point.",
                "A collection of these averages forms a new distribution."
            ]
        )
        
        # Colors for lecture lines
        colors = [TEAL, YELLOW, ORANGE]

        # === Animation for Lecture Line 1 ===
        # "We repeat this sampling process thousands of times."
        self.play(self.lecture[0].animate.set_color(colors[0]))

        # Visual: Conveyor belt at Row C (C1 to C6)
        belt = Line(self.grid["C1"] + LEFT*0.5, self.grid["C6"] + RIGHT*0.5, color=GREY_B)
        
        # belt_label: Addressing Issue 28 - centering above belt.
        # Positioned at Row B (1 unit away from belt at Row C)
        belt_label = Text("Conveyor Belt", font_size=16, color=GREY_B)
        self.place_in_area(belt_label, "B3", "B4", scale_factor=0.8)
        
        # Create a "Group of 30" representation
        def create_monster_group():
            monsters = VGroup(*[Dot(radius=0.04, color=BLUE) for _ in range(15)])
            monsters.arrange_in_grid(rows=3, cols=5, buff=0.04)
            label = Text("n=30", font_size=12, color=BLUE).next_to(monsters, UP, buff=0.1)
            return VGroup(monsters, label)

        # Addressing Issue 27: Use full width of belt by moving groups across.
        group1 = create_monster_group()
        group2 = create_monster_group()
        
        # Start positions on Row C
        self.place_at_grid(group1, "C1")
        # group2 starts off-screen left
        group2.move_to(self.grid["C1"] + LEFT * 2)

        self.add(belt, belt_label)
        
        # Animate movement of groups along the belt
        self.play(
            group1.animate.move_to(self.grid["C3"]),
            group2.animate.move_to(self.grid["C1"]),
            run_time=2,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "Each group's average is recorded as a single data point."
        self.play(self.lecture[1].animate.set_color(colors[1]))

        # Number line at Row E (as per storyboard)
        number_line = NumberLine(
            x_range=[0, 10, 1],
            length=5,
            include_numbers=True,
            font_size=16,
            color=WHITE
        )
        self.place_in_area(number_line, "E1", "E6")
        
        # nl_label: Addressing Issue 29 - centering above the number line area.
        # Positioned at Row D (1 unit away from number line at Row E)
        nl_label = Text("Sampling Distribution of Means", font_size=16, color=WHITE)
        self.place_in_area(nl_label, "D4", "D6", scale_factor=0.8)
        
        self.add(number_line, nl_label)

        # For the first group (at C3), drop a mean dot to Row E
        mean_dot1 = Dot(color=colors[1], radius=0.08)
        mean_dot1.move_to(group1.get_center())
        
        target_pos1 = number_line.n2p(5) # Theoretical average height
        
        # Move groups further right while dropping the first mean dot
        self.play(
            group1.animate.move_to(self.grid["C5"]),
            group2.animate.move_to(self.grid["C3"]),
            FadeIn(mean_dot1),
            mean_dot1.animate.move_to(target_pos1),
            run_time=1.5
        )
        
        # Drop dot for the second group now at the center
        mean_dot2 = Dot(color=colors[1], radius=0.08)
        mean_dot2.move_to(group2.get_center())
        target_pos2 = number_line.n2p(6) # Another sample average
        
        self.play(
            group2.animate.move_to(self.grid["C5"]),
            FadeIn(mean_dot2),
            mean_dot2.animate.move_to(target_pos2),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "A collection of these averages forms a new distribution."
        self.play(self.lecture[2].animate.set_color(colors[2]))

        # Rapidly repeat dot dropping from the belt center (C3-C4) to form a histogram shape
        def get_drop_animation(val, offset_y=0):
            d = Dot(color=colors[2], radius=0.06)
            # Start from the center of the belt area
            belt_center = (self.grid["C3"] + self.grid["C4"]) / 2
            d.move_to(belt_center)
            target = number_line.n2p(val) + UP * offset_y
            return AnimationGroup(FadeIn(d), d.animate.move_to(target))

        # Sample values to visually approximate a normal distribution
        sample_values = [5, 4, 6, 5, 5, 4, 6, 7, 3, 5, 5, 6, 4, 5, 4, 6, 5, 5, 5]
        stacks = {v: 0 for v in range(11)}
        
        all_drops = []
        for val in sample_values:
            # Check stacks to avoid colliding with Row D label (offset_y < 1.0)
            drop_anim = get_drop_animation(val, offset_y=stacks[val]*0.1)
            all_drops.append(drop_anim)
            stacks[val] += 1

        # Play animations in rapid batches
        self.play(AnimationGroup(*all_drops[:5], lag_ratio=0.2))
        self.play(AnimationGroup(*all_drops[5:12], lag_ratio=0.1))
        self.play(AnimationGroup(*all_drops[12:], lag_ratio=0.05))

        self.wait(2)
