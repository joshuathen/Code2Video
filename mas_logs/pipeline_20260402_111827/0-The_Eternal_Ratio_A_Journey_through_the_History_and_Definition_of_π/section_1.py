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
        # Setup the layout with provided lecture lines
        title_str = "The Hook: Penny the Panda's Hula Hoop"
        lines_str = [
            "Meet Penny the Panda and her yellow hula hoop.",
            "We can unroll this hoop into a straight line.",
            "She has a larger orange hoop that unrolls too.",
            "Compare both unrolled lengths to their circle's diameter.",
            "Each line is just over three diameters long."
        ]
        self.setup_layout(title_str, lines_str)

        # --- Assets Construction ---
        # Penny the Panda Asset
        panda = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/panda.svg")
        
        # Small Yellow Hoop
        small_radius = 0.4
        small_hoop = Circle(radius=small_radius, color="#FFFF00", stroke_width=4)
        
        # Medium Orange Hoop
        med_radius = 0.6
        med_hoop = Circle(radius=med_radius, color="#FFA500", stroke_width=5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        # Issue 23: panda at B2
        self.place_at_grid(panda, 'B2', scale_factor=0.8)
        # Issue 24: small_hoop at C3
        self.place_at_grid(small_hoop, 'C3', scale_factor=0.8)
        self.play(FadeIn(panda), Create(small_hoop))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Unrolling small hoop
        # The hoop is at grid position C3. We unroll to the right.
        small_start_pos = small_hoop.get_center()
        small_unrolled_line = Line(
            small_start_pos + DOWN * small_hoop.height/2,
            small_start_pos + DOWN * small_hoop.height/2 + RIGHT * (2 * PI * small_radius * 0.8),
            color="#FFFF00"
        )
        
        # To simulate unrolling, we rotate the circle and move it while the line grows
        self.play(
            small_hoop.animate.shift(RIGHT * (2 * PI * small_radius * 0.8)).rotate(-2 * PI),
            Create(small_unrolled_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFA500")
        
        # Issue 25: med_hoop at E3
        self.place_at_grid(med_hoop, 'E3', scale_factor=0.8)
        
        med_start_pos = med_hoop.get_center()
        med_unrolled_line = Line(
            med_start_pos + DOWN * med_hoop.height/2,
            med_start_pos + DOWN * med_hoop.height/2 + RIGHT * (2 * PI * med_radius * 0.8),
            color="#FFA500"
        )
        
        self.play(Create(med_hoop))
        self.play(
            med_hoop.animate.shift(RIGHT * (2 * PI * med_radius * 0.8)).rotate(-2 * PI),
            Create(med_unrolled_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFFFF")
        
        # Create diameters
        diam_small = Line(
            small_start_pos + LEFT * small_radius * 0.8,
            small_start_pos + RIGHT * small_radius * 0.8,
            color="#FFFFFF"
        ).shift(DOWN * (small_radius * 0.8 + 0.3))
        
        diam_med = Line(
            med_start_pos + LEFT * med_radius * 0.8,
            med_start_pos + RIGHT * med_radius * 0.8,
            color="#FFFFFF"
        ).shift(DOWN * (med_radius * 0.8 + 0.3))
        
        label_small = Text("Diameter", font_size=16, color="#FFFFFF").next_to(diam_small, DOWN, buff=0.1)
        label_med = Text("Diameter", font_size=16, color="#FFFFFF").next_to(diam_med, DOWN, buff=0.1)

        self.play(
            Create(diam_small), Write(label_small),
            Create(diam_med), Write(label_med)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF")
        
        # Place three diameters along each line
        def get_diams(line, diam_len, color):
            diams = VGroup()
            start = line.get_start()
            for i in range(3):
                d = Line(
                    start + RIGHT * i * diam_len,
                    start + RIGHT * (i+1) * diam_len,
                    color=color, stroke_width=6
                ).shift(UP * 0.1)
                diams.add(d)
            return diams

        small_diams = get_diams(small_unrolled_line, 2 * small_radius * 0.8, WHITE)
        med_diams = get_diams(med_unrolled_line, 2 * med_radius * 0.8, WHITE)

        self.play(FadeIn(small_diams), FadeIn(med_diams))
        self.wait(2)
