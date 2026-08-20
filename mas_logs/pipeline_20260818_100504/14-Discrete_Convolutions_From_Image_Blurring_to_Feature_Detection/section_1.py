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
        lecture_lines = [
            "Convolution starts with a local neighborhood.",
            "Imagine a sliding window over pixels.",
            "This window computes a local weighted average.",
            "It smooths or highlights specific image features.",
            "This process is the discrete convolution."
        ]
        self.setup_layout("Intuitive Hook: The 'Moving Window'", lecture_lines)
        
        # Load Assets
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        photo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")

        # === Animation for Lecture Line 1 ===
        # Create a 1D sequence
        sequence = VGroup(*[Square(side_length=0.6).set_fill(BLUE, opacity=0.5).add(Integer(i)) for i in range(5)])
        sequence.arrange(RIGHT, buff=0.1)
        self.place_in_area(sequence, 'B1', 'B5', scale_factor=0.7)
        self.add(sequence)
        
        self.place_at_grid(camera_icon, 'A3', scale_factor=0.5)
        self.play(FadeIn(camera_icon))
        
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        window = Rectangle(width=1.9, height=0.7, color=RED, stroke_width=4)
        self.place_at_grid(window, 'C3', scale_factor=0.8) # Adjusted per issue 18
        self.add(window)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        calc_text = Text("Sum = 1*w1 + 2*w2 + 3*w3", font_size=20)
        self.place_at_grid(calc_text, 'D2', scale_factor=0.9) # Adjusted per issue 17
        self.play(Write(calc_text))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        result = Integer(6)
        self.place_at_grid(result, 'E4', scale_factor=1.0) # Adjusted per issue 19
        self.play(FadeIn(result))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        self.play(window.animate.shift(RIGHT * 0.7))
        self.place_at_grid(photo_icon, 'F4', scale_factor=0.5)
        self.play(FadeIn(photo_icon))
        
        self.lecture[4].set_color(YELLOW)
        self.wait(1)
