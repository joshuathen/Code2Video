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
        self.setup_layout("Prerequisite: The Forward Pass", ["Data flows through weighted connections.", "Each edge acts as a multiplier.", "The final output is a prediction."])
        
        # Title Asset
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        self.place_at_grid(computer_icon, "A5", scale_factor=0.3)
        self.play(FadeIn(computer_icon))
        
        # Microchip Asset
        microchip_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg", color=GREY)
        
        # Arrange layers in B and D rows as requested
        layer1 = VGroup(*[Circle(radius=0.2, color="#00FFFF", fill_opacity=0.3) for _ in range(3)]).arrange(DOWN, buff=0.4)
        layer2 = VGroup(*[Circle(radius=0.2, color="#00FFFF", fill_opacity=0.3) for _ in range(3)]).arrange(DOWN, buff=0.4)
        
        self.place_in_area(layer1, 'B2', 'B3', scale_factor=0.7)
        self.place_in_area(layer2, 'D2', 'D3', scale_factor=0.7)
        
        weights = VGroup()
        for n1 in layer1:
            for n2 in layer2:
                line = Line(n1.get_center(), n2.get_center(), stroke_width=2, color=GREY)
                weights.add(line)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(FadeIn(layer1), FadeIn(layer2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.place_at_grid(microchip_icon, 'C3', scale_factor=0.5)
        self.play(Create(weights), FadeIn(microchip_icon))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        prediction = Text("Y_hat", color="#FFFF00", font_size=24)
        self.place_at_grid(prediction, 'C5', scale_factor=0.9)
        self.play(Write(prediction))
        self.wait(2)
