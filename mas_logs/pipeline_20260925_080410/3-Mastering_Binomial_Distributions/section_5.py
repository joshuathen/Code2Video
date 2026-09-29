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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application: Quality Control", [
            "Binomial distribution applies in industry.",
            "Use for quality control tasks.",
            "Independence is crucial check."
        ])
        
        # Define objects
        batch = VGroup(*[Circle(radius=0.2, color=BLUE, fill_opacity=0.5) for _ in range(10)])
        batch.arrange(RIGHT, buff=0.1)
        
        # Assets
        factory_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg")
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.place_in_area(batch, 'A4', 'B6', scale_factor=0.6)
        self.place_at_grid(factory_icon, 'A2', scale_factor=0.5)
        factory_icon.set_color("#FF0000")
        
        self.play(Create(batch), FadeIn(factory_icon))
        
        # Defect simulation
        defects = [2, 5, 8]
        for idx in defects:
            batch[idx].set_color("#FF0000")
        self.play(*[Flash(batch[i], color="#FF0000") for i in defects])

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        prob_label = MathTex(r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}").set_color(WHITE)
        self.place_at_grid(prob_label, 'D1', scale_factor=0.7)
        self.play(Write(prob_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        boundary = Rectangle(width=4, height=1, color="#00FF00").set_fill("#00FF00", opacity=0.2)
        status = Text("Accept Batch", font_size=24, color="#00FF00")
        
        self.place_at_grid(boundary, 'D4', scale_factor=0.7)
        self.place_at_grid(status, 'E6', scale_factor=0.7)
        self.place_at_grid(sensor_icon, 'D6', scale_factor=0.4)
        sensor_icon.set_color("#00FF00")
        
        self.play(FadeIn(boundary), FadeIn(status), FadeIn(sensor_icon))
        self.wait(2)
