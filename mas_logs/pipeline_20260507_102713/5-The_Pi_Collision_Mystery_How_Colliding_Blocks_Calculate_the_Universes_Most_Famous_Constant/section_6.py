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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup basic layout with specific lecture lines
        lecture_lines = [
            "Phase space reveals hidden constants in nature.",
            "Pi emerges from the symmetry of conservation.",
            "Abstract geometry solves deep physical mysteries."
        ]
        self.setup_layout("Conclusion: The Power of Phase Space", lecture_lines)

        # Colors for the lines
        COLOR_L1 = "#87CEEB" # Sky Blue
        COLOR_L2 = "#FFFF00" # Yellow
        COLOR_L3 = "#FFD700" # Gold

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_L1)
        
        # Simulation environment (Left Side)
        wall = Line(self.grid["B1"] + LEFT*0.3, self.grid["E1"] + LEFT*0.3, color=GREY)
        floor = Line(self.grid["E1"] + LEFT*0.3, self.grid["E3"] + RIGHT*0.3, color=GREY)
        
        # Use provided asset for blocks
        blocks_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/blocks.svg")
        self.place_in_area(blocks_svg, "C1", "E3", scale_factor=1.0)
        
        # Phase Space Axes (Right Side)
        axes = Axes(
            x_range=[-1.2, 1.2, 1], y_range=[-1.2, 1.2, 1],
            x_length=3, y_length=3,
            tips=False,
            axis_config={"color": GREY_C}
        )
        self.place_in_area(axes, "B4", "E6")
        
        phase_label = Text("Phase Space", font_size=20, color=COLOR_L1)
        self.place_in_area(phase_label, "A4", "A6", scale_factor=0.8)
        
        # Phase Space State Point and Path
        state_point = Dot(axes.c2p(1, 0), color=COLOR_L1)
        circle_path = Arc(radius=axes.get_x_unit_size(), start_angle=0, angle=0, color=COLOR_L1)
        circle_path.move_to(axes.c2p(0, 0))

        self.play(
            Create(wall), Create(floor),
            FadeIn(blocks_svg),
            Create(axes),
            Write(phase_label),
            FadeIn(state_point)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_L2)
        
        # Tracker for collisions (M = 100^3 -> 3141 collisions)
        collision_tracker = ValueTracker(0)
        
        # Collision counter elements
        counter_label = Text("Collisions:", font_size=20, color=COLOR_L2)
        self.place_at_grid(counter_label, "A2")
        
        counter_num = DecimalNumber(0, num_decimal_places=0, color=COLOR_L2, mob_class=Text)
        self.place_at_grid(counter_num, "A3")
        
        # Dynamic Updaters
        counter_num.add_updater(lambda d: d.set_value(collision_tracker.get_value()))
        
        def update_arc(obj):
            angle = -PI * (collision_tracker.get_value() / 3141)
            # Efficiently update arc geometry
            new_arc = Arc(
                radius=axes.get_x_unit_size(), 
                start_angle=0, 
                angle=angle, 
                color=COLOR_L2
            ).move_to(axes.c2p(0,0))
            obj.become(new_arc)

        def update_point(obj):
            angle = -PI * (collision_tracker.get_value() / 3141)
            obj.move_to(axes.c2p(np.cos(angle), np.sin(angle)))
            
        def update_blocks(obj):
            # Visual buzz representing high-frequency collisions
            val = collision_tracker.get_value()
            if val > 0:
                obj.set_x(self.grid["D2"][0] + np.sin(val * 0.5) * 0.05)

        circle_path.add_updater(update_arc)
        state_point.add_updater(update_point)
        blocks_svg.add_updater(update_blocks)

        self.add(counter_num, circle_path)
        
        # Run simulation to 3141
        self.play(
            collision_tracker.animate.set_value(3141),
            run_time=8,
            rate_func=linear
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_L3)
        
        # Final Gold Summary Text
        final_text = Text("The geometry of phase space\nreveals pi", font_size=32, color=COLOR_L3)
        self.place_in_area(final_text, "C2", "E5", scale_factor=1.0)
        
        # Clear screen of previous elements
        self.play(
            FadeOut(wall), FadeOut(floor),
            FadeOut(blocks_svg),
            FadeOut(axes),
            FadeOut(circle_path),
            FadeOut(state_point),
            FadeOut(counter_label),
            FadeOut(counter_num),
            FadeOut(phase_label)
        )
        
        # Final impact message
        self.play(Write(final_text))
        self.wait(3)
