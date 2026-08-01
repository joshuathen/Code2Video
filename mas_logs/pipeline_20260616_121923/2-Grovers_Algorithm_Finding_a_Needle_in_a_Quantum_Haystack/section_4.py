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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initializing scene with Title and Lecture Lines
        self.setup_layout(
            "Step 2: The Grover Diffusion Operator", 
            [
                'Next, we apply the Grover diffusion operator.', 
                'First, we calculate the average height of all bars.', 
                'We then flip every bar across this average line.', 
                'The target amplitude grows significantly larger than others.', 
                'Non-target states shrink closer to a zero value.'
            ]
        )

        # === Data and Math Setup ===
        # Initial heights: 7 white bars at +1.0, 1 gold bar at -1.0 (after Oracle flip)
        # Average (Mean): (7 * 1.0 + 1 * -1.0) / 8 = 6 / 8 = 0.75
        # Inversion formula: new_h = 2 * mean - old_h
        # White new: 2 * 0.75 - 1.0 = 0.5
        # Gold new: 2 * 0.75 - (-1.0) = 2.5
        
        bar_width = 0.4
        spacing = 0.1
        target_index = 4
        
        # Colors
        WHITE_COLOR = "#FFFFFF"
        GOLD_COLOR = "#FFD700"
        MEAN_COLOR = "#00FF00"
        
        # Create X-axis
        axis = Line(LEFT * 2.5, RIGHT * 2.5, color=GREY_B)
        
        # Create initial bars
        initial_bars = VGroup()
        for i in range(8):
            color = WHITE_COLOR if i != target_index else GOLD_COLOR
            h = 1.0 if i != target_index else -1.0
            
            # Create rectangle
            rect = Rectangle(
                width=bar_width, 
                height=abs(h), 
                fill_color=color, 
                fill_opacity=0.8, 
                stroke_width=1
            )
            # Position relative to baseline
            if h > 0:
                rect.next_to(axis.get_start() + RIGHT * (i * (bar_width + spacing) + 0.3), UP, buff=0)
            else:
                rect.next_to(axis.get_start() + RIGHT * (i * (bar_width + spacing) + 0.3), DOWN, buff=0)
            initial_bars.add(rect)
            
        visualization = VGroup(axis, initial_bars)
        # Fix for Issue 35, 36, 37: 
        # B2-F5 provides vertical margin from title (Row A) and horizontal margin from right edge (Col 6).
        # Scale 0.75 provides enough room for the bar growth.
        self.place_in_area(visualization, 'B2', 'F5', scale_factor=0.75)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(axis), Create(initial_bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Calculate mean line position relative to visualization
        # Heights are 1.0 unit. We need to find the pixel height of 0.75 units.
        y_origin = axis.get_center()[1]
        y_unit_size = initial_bars[0].height # The height of a white bar is 1.0 unit
        mean_y = y_origin + (0.75 * y_unit_size)
        
        mean_line = DashedLine(
            start=[axis.get_left()[0], mean_y, 0],
            end=[axis.get_right()[0], mean_y, 0],
            color=MEAN_COLOR,
            stroke_width=4
        )
        mean_label = Text("Mean", font_size=16, color=MEAN_COLOR).next_to(mean_line, RIGHT, buff=0.1)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Prepare targets for transform
        new_bars = VGroup()
        for i in range(8):
            color = WHITE_COLOR if i != target_index else GOLD_COLOR
            new_h_val = 0.5 if i != target_index else 2.5
            
            # Create new rectangle with updated height
            new_rect = Rectangle(
                width=bar_width, 
                height=new_h_val * y_unit_size, 
                fill_color=color, 
                fill_opacity=0.8, 
                stroke_width=1
            )
            # All bars are now positive after inversion (since mean 0.75 > 0.5 and 2.5)
            new_rect.next_to(axis.get_start() + RIGHT * (i * (bar_width + spacing) + 0.3), UP, buff=0)
            new_bars.add(new_rect)
            
        self.play(Transform(initial_bars, new_bars), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        target_bar = initial_bars[target_index]
        self.play(
            target_bar.animate.set_fill(opacity=1.0),
            Flash(target_bar, color=GOLD_COLOR, line_length=0.3, num_lines=12)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        non_target_anims = []
        for i, bar in enumerate(initial_bars):
            if i != target_index:
                non_target_anims.append(bar.animate.set_fill(opacity=0.3))
        
        self.play(*non_target_anims)
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
