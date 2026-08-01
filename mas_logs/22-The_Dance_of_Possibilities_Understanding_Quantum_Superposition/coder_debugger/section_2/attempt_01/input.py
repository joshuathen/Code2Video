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
        lecture_lines = [
            'In the quantum world, objects act very differently.',
            'Meet Pixel, a cat in a blurry cloud of possibilities.',
            'She exists on the left and right sides simultaneously.',
            'This dynamic state is called a quantum superposition.',
            'Pixel represents multiple potential outcomes existing at once.'
        ]
        self.setup_layout("The Quantum Identity Crisis", lecture_lines)

        # Pre-define colors
        WHITE_COLOR = "#FFFFFF"
        GRAY_COLOR = "#808080"
        PLATFORM_COLOR = "#AAAAAA"
        QUANTUM_COLOR = "#00FFFF"

        # Define Objects
        platform = Rectangle(width=4.5, height=0.6, fill_color=PLATFORM_COLOR, fill_opacity=0.8, stroke_color=WHITE)
        self.place_in_area(platform, "C2", "D5", scale_factor=0.8) # Issue 30: scale_factor=0.8

        left_label = Text("Left", font_size=24, color=WHITE_COLOR)
        self.place_at_grid(left_label, "B2")
        
        right_label = Text("Right", font_size=24, color=WHITE_COLOR)
        self.place_at_grid(right_label, "B5")

        # Asset Cat (Pixel) - Issue 24: Asset path integration
        pixel_cat = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png")
        # Issue 29: Place in area C2-D5 so it spans the platform
        self.place_in_area(pixel_cat, "C2", "D5", scale_factor=0.6) 
        # Tint if possible (ImageMobject set_color tints the image)
        pixel_cat.set_color(WHITE_COLOR)

        # Superposition Cloud
        cloud = Ellipse(width=5.0, height=1.5, fill_color=QUANTUM_COLOR, fill_opacity=0.5, stroke_width=0)
        self.place_in_area(cloud, "C2", "D5")

        # State label - Issue 31: Moved to E2-E5
        state_text = Text("State of Superposition", font_size=28, color=WHITE_COLOR)
        self.place_in_area(state_text, "E2", "E5")

        # === Animation for Lecture Line 1 ===
        # 'In the quantum world, objects act very differently.'
        self.play(self.lecture[0].animate.set_color(WHITE_COLOR))
        self.play(Create(platform), Write(left_label), Write(right_label))
        self.play(FadeIn(pixel_cat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'Meet Pixel, a cat in a blurry cloud of possibilities.'
        self.play(self.lecture[0].animate.set_color(GRAY_COLOR), self.lecture[1].animate.set_color(QUANTUM_COLOR))
        # Cat fades into cloud
        self.play(
            FadeOut(pixel_cat),
            FadeIn(cloud, scale=0.8),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'She exists on the left and right sides simultaneously.'
        self.play(self.lecture[1].animate.set_color(GRAY_COLOR), self.lecture[2].animate.set_color(QUANTUM_COLOR))
        self.play(
            left_label.animate.set_color(QUANTUM_COLOR),
            right_label.animate.set_color(QUANTUM_COLOR),
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'This dynamic state is called a quantum superposition.'
        self.play(self.lecture[2].animate.set_color(GRAY_COLOR), self.lecture[3].animate.set_color(QUANTUM_COLOR))
        # Pulsing cloud animation
        self.play(
            cloud.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.play(
            cloud.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'Pixel represents multiple potential outcomes existing at once.'
        self.play(self.lecture[3].animate.set_color(GRAY_COLOR), self.lecture[4].animate.set_color(WHITE_COLOR))
        self.play(Write(state_text))
        self.wait(2)
