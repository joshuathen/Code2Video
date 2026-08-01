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
        # Setup layout with title and specific lecture lines
        self.setup_layout(
            "Prerequisite: The Complex Plane as a Map", 
            [
                "Welcome to the complex plane, a 2D number map.", 
                "We begin at one on the real axis.", 
                "Our guide, Digit, will lead the way."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight line 1
        self.lecture[0].set_color(YELLOW)
        
        # 1. Fade in a 2D coordinate system with white (#FFFFFF) axes and labels 'Real' and 'Imaginary'.
        axes = Axes(
            x_range=[-2.2, 2.2, 1],
            y_range=[-2.2, 2.2, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": "#FFFFFF", "include_tip": True},
            tips=True
        )
        # Fix Issue 39: Scale axes and place in area
        self.place_in_area(axes, 'A1', 'F6', scale_factor=0.8)
        
        real_label = Text("Real", font_size=16, color="#FFFFFF")
        imag_label = Text("Imaginary", font_size=16, color="#FFFFFF")
        
        # Position labels within 1 grid unit of corresponding objects (near axis tips)
        real_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.1)
        imag_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.1)
        
        self.play(
            Create(axes),
            Write(real_label),
            Write(imag_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Switch focus to lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # 2. Highlight the horizontal 'Real' axis in blue (#0000FF) and vertical 'Imaginary' in red (#FF0000)
        self.play(
            axes.x_axis.animate.set_color("#0000FF"),
            real_label.animate.set_color("#0000FF"),
            run_time=1
        )
        self.play(
            axes.y_axis.animate.set_color("#FF0000"),
            imag_label.animate.set_color("#FF0000"),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch focus to lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # 3. Draw a unit circle in white (#FFFFFF) centered at the origin
        unit_circle = Circle(radius=axes.x_axis.get_unit_size(), color="#FFFFFF").move_to(axes.get_center())
        
        # Load Digit SVG Asset (Issue 37)
        digit = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/digit.svg")
        # Fix Issue 40: Place Digit at grid D5
        self.place_at_grid(digit, 'D5', scale_factor=0.5)
        
        # Create numeric label '1'
        label_one = Text("1", font_size=18, color="#FFFFFF")
        # Fix Issue 41: Place label '1' at grid E5
        self.place_at_grid(label_one, 'E5', scale_factor=0.8)
        
        self.play(Create(unit_circle), run_time=1.5)
        self.play(
            FadeIn(digit),
            Write(label_one),
            run_time=1
        )
        self.wait(3)
