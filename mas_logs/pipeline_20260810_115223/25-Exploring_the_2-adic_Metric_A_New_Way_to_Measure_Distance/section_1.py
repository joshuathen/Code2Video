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
        self.setup_layout("Prerequisite: The Usual Distance", [
            "Standard distance makes small numbers near zero.", 
            "Euclidean metrics show 1/8 as very tiny.", 
            "A rabbit hops closer to zero point."
        ])
        
        # Elements
        number_line = NumberLine(x_range=[-1, 5, 1], length=5, include_numbers=True).set_color(WHITE)
        # Apply fix for Issue 25/40
        self.place_in_area(number_line, 'A2', 'C5', scale_factor=0.9)
        
        rabbit_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg")
        # Apply fix for Issue 27/40
        self.place_at_grid(rabbit_icon, 'C2', scale_factor=1.2)
        
        origin_label = Dot(number_line.n2p(0), color=WHITE)
        origin_text = Text("0", font_size=20).next_to(origin_label, DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(number_line), Write(origin_label), Write(origin_text), FadeIn(rabbit_icon))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        points = [2, 4, 8, 16]
        dots = VGroup(*[Dot(number_line.n2p(p/4), color="#FF0000") for p in points])
        labels = VGroup(*[Text(f"1/{p}", font_size=16).next_to(d, UP) for d, p in zip(dots, points)])
        
        # Apply fix for Issue 26/40
        self.place_at_grid(labels, 'B2', scale_factor=0.7)
        
        self.play(FadeIn(dots), FadeIn(labels))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        self.play(rabbit_icon.animate.move_to(number_line.n2p(0.25)), run_time=2)
        self.lecture[2].set_color("#00FF00")
        
        # Fade out labels as per storyboard
        self.play(FadeOut(labels), FadeOut(origin_text))
        
        self.wait(1)
