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
        title = "The Power of Abstraction: Real-world AI"
        lecture_lines = [
            "Artificial intelligence treats data points as abstract vectors.",
            "Word embeddings map meanings into high-dimensional space.",
            "Vector math helps machines understand similarities in data."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        waveform_color = "#00FF00"
        
        # Asset: Microphone [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg]
        # Resolve Issue 22
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        mic.set_color(WHITE)
        self.place_at_grid(mic, "B1", scale_factor=0.5)
        
        # Waveform setup - Resolve Issue 32
        waveform = FunctionGraph(
            lambda x: 0.3 * np.sin(5 * x) * np.exp(-0.5 * x**2),
            x_range=[-1.5, 1.5],
            color=waveform_color
        )
        self.place_in_area(waveform, 'B1', 'C2', scale_factor=0.8)
        
        # Vector setup - Resolve Issue 33
        vector_box = MathTex(
            r"\begin{bmatrix} 0.12 \\ -0.45 \\ 0.89 \\ 0.03 \\ -0.71 \end{bmatrix}",
            color=waveform_color
        )
        self.place_in_area(vector_box, 'B4', 'C5', scale_factor=0.8)

        # Flow arrow from waveform area to vector area
        flow_arrow = Arrow(
            start=self.grid["B2"], 
            end=self.grid["B4"], 
            color=WHITE, 
            buff=0.3
        )

        # Execution Line 1
        self.play(self.lecture[0].animate.set_color(waveform_color))
        self.play(FadeIn(mic))
        self.play(Create(waveform))
        self.play(GrowArrow(flow_arrow))
        self.play(Write(vector_box))
        self.wait(1)
        self.play(FadeOut(waveform), FadeOut(vector_box), FadeOut(flow_arrow), FadeOut(mic))

        # === Animation for Lecture Line 2 ===
        embedding_color = "#FFFF00"
        
        # 2D Cloud of points
        np.random.seed(42)
        points = []
        for _ in range(20):
            # Randomly distribute points centered around E4
            x_offset = np.random.uniform(-1.5, 1.5)
            y_offset = np.random.uniform(-1, 1)
            p = Dot(radius=0.05, color=BLUE_B, fill_opacity=0.6)
            p.move_to(self.grid["E4"] + np.array([x_offset, y_offset, 0]))
            points.append(p)
        cloud = VGroup(*points)
        
        # 'Input' point - Resolve Issue 34
        input_dot = Dot(radius=0.1, color=embedding_color)
        self.place_at_grid(input_dot, 'E3', scale_factor=1.0)
        
        input_label = Text("Input", font_size=16, color=embedding_color)
        input_label.next_to(input_dot, UP, buff=0.1)

        # Execution Line 2
        self.play(self.lecture[1].animate.set_color(embedding_color))
        self.play(FadeIn(cloud))
        self.play(FadeIn(input_dot), Write(input_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        math_color = "#FFFFFF"
        
        # Identify a match (predetermined from seed)
        target_point = points[5]
        
        match_line = Line(
            start=input_dot.get_center(),
            end=target_point.get_center(),
            color=math_color,
            stroke_width=2
        )
        
        match_label = Text("Matches", font_size=16, color=WHITE)
        match_label.next_to(target_point, DOWN, buff=0.1)

        # Execution Line 3
        self.play(self.lecture[2].animate.set_color(math_color))
        self.play(Create(match_line))
        self.play(
            target_point.animate.scale(2).set_color(WHITE),
            Write(match_label)
        )
        
        # Glow effect
        glow = Circle(radius=0.3, color=WHITE, stroke_width=2).move_to(target_point)
        self.play(ScaleInPlace(glow, 2), FadeOut(glow), run_time=1)
        
        self.wait(2)
