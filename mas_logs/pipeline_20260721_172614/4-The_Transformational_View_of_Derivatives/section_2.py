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
        # Data from storyboard
        title_text = "Prerequisite: Linear Transformations on 1D Space"
        lecture_lines = [
            "Consider a simple linear transformation like y equals mx.",
            "The constant m acts as a uniform scaling factor.",
            "It stretches or squashes the entire input number line."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_LINE = "#FFFFFF"
        COLOR_LABEL = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # "Consider a simple linear transformation like y equals mx."
        self.lecture[0].set_color(YELLOW)
        
        # Construction of the Number Line
        ticks = VGroup()
        tick_labels = VGroup()
        initial_gap = 0.6
        for i in range(3):
            # Ticks at 0, 1, 2 initially spaced
            pos = i * initial_gap
            tick = Line(UP*0.1, DOWN*0.1, color=COLOR_LINE).move_to(RIGHT * pos)
            label = Text(str(i), font_size=20, color=COLOR_LINE).next_to(tick, DOWN, buff=0.2)
            ticks.add(tick)
            tick_labels.add(label)
        
        # Line spans slightly beyond the ticks
        line_main = Line(start=LEFT*0.3, end=RIGHT*(2 * initial_gap + 0.3), color=COLOR_LINE)
        # Shift line so it starts relative to tick 0
        line_main.move_to(ticks[0].get_center() + RIGHT * (initial_gap), coor_mask=np.array([1, 0, 0]))

        number_line = VGroup(line_main, ticks, tick_labels)
        
        # Positioning: As per issue 24, place in area C3 to D6
        self.place_in_area(number_line, "C3", "D6", scale_factor=1.0)
        
        # Alignment: Ensure tick 0 starts at grid C3
        c3_pos = self.grid["C3"]
        # We want the '0' tick to stay relatively stable.
        number_line.shift(c3_pos - ticks[0].get_center())
        
        self.play(Create(number_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The constant m acts as a uniform scaling factor."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        m_label = Text("m = 3", font_size=28, color=COLOR_LABEL)
        text_scaling = Text("Constant Scaling Factor", font_size=22, color=COLOR_LABEL)
        label_group = VGroup(m_label, text_scaling).arrange(DOWN, buff=0.2)
        
        # As per issue 23, place label group in area A2 to A5
        self.place_in_area(label_group, "A2", "A5", scale_factor=0.8)
        
        self.play(Write(label_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "It stretches or squashes the entire input number line."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Setup ValueTracker for expansion (scaling factor m)
        stretch_factor = ValueTracker(1.0)
        
        # Save fixed position parameters for the updater
        fixed_pos_0 = ticks[0].get_center().copy()
        base_y = fixed_pos_0[1]
        fixed_x = fixed_pos_0[0]
        
        # Relative boundaries for the line
        rel_start = -0.3
        rel_end = (2 * initial_gap) + 0.3

        def update_number_line(obj):
            s = stretch_factor.get_value()
            # Update ticks and labels relative to tick 0
            for i in range(3):
                new_x = fixed_x + i * initial_gap * s
                ticks[i].set_x(new_x)
                tick_labels[i].set_x(new_x)
            
            # Update line start and end relative to tick 0
            new_start_x = fixed_x + rel_start * s
            new_end_x = fixed_x + rel_end * s
            line_main.set_points_as_corners([
                [new_start_x, base_y, 0],
                [new_end_x, base_y, 0]
            ])

        number_line.add_updater(update_number_line)
        
        # Animate the stretch from factor 1 to factor 3
        # Storyboard: "Animate the line expanding so the distance between ticks triples."
        self.play(stretch_factor.animate.set_value(3.0), run_time=3)
        self.wait(2)
        
        # Cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(2)
