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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        self.setup_layout(
            "Application: Designing a Cooling System",
            [
                "Engineers use PDEs to optimize cooling for electronics.",
                "We simulate heat flowing away from a hot CPU.",
                "This prevents overheating before a physical prototype exists."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Engineers use PDEs to optimize cooling for electronics.
        self.lecture[0].set_color(YELLOW)
        
        # CPU Mobject
        cpu = Square(side_length=1.2, fill_opacity=0.8, fill_color="#FF4500", stroke_color=WHITE)
        self.place_at_grid(cpu, "C2")
        
        cpu_label = Text("CPU", font_size=20)
        self.place_at_grid(cpu_label, "B2")
        
        # Heat pipe
        pipe = Rectangle(width=2.5, height=0.6, fill_opacity=0.6, fill_color="#0000FF", stroke_color=WHITE)
        self.place_at_grid(pipe, "C4") # Resolved Issue 30: Moved from C5 to C4
        
        pipe_label = Text("Heat Pipe", font_size=20)
        self.place_at_grid(pipe_label, "B4")
        
        # Heat lines (static indicators at first)
        heat_lines = VGroup(*[
            Line(cpu.get_right() + UP*0.2*i, cpu.get_right() + RIGHT*0.4 + UP*0.2*i, color="#FF0000", stroke_width=2)
            for i in range(-2, 3)
        ])

        self.play(
            FadeIn(cpu),
            FadeIn(cpu_label),
            FadeIn(pipe),
            FadeIn(pipe_label),
            Create(heat_lines)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We simulate heat flowing away from a hot CPU.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Flow particles (representing heat moving into the pipe)
        particles = VGroup(*[
            Dot(radius=0.06, color="#FF0000") for _ in range(10)
        ])
        
        # Initial positions at the edge of the CPU
        for i, p in enumerate(particles):
            p.move_to(cpu.get_right() + UP * (0.8 * (i/10.0 - 0.5)))

        self.play(FadeIn(particles))
        
        # Animate flow to the heat pipe
        flow_animations = []
        for i, p in enumerate(particles):
            target_pos = pipe.get_left() + UP * (0.4 * (i/10.0 - 0.5))
            flow_animations.append(p.animate(run_time=2, rate_func=linear).move_to(target_pos))
        
        self.play(*flow_animations)
        self.play(FadeOut(particles), FadeOut(heat_lines))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This prevents overheating before a physical prototype exists.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Transition the CPU color from hot red/orange to cool blue
        self.play(
            cpu.animate.set_fill("#0000FF"),
            cpu_label.animate.set_color(BLUE),
            run_time=2
        )
        
        # Optimized indicator
        optimized_text = Text("Optimized!", color=GREEN, font_size=24)
        self.place_at_grid(optimized_text, "E2") # Resolved Issue 31: Moved from E4 to E2
        
        self.play(Write(optimized_text))
        self.wait(2)
