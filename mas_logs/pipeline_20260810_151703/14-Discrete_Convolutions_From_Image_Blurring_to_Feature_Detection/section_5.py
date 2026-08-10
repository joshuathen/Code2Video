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
        self.setup_layout("Real-World Impact: Neural Networks", [
            "CNNs learn these filter patterns automatically.",
            "Edges form the foundation for features.",
            "Filters stack to recognize complex objects."
        ])
        
        # Assets
        building_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/building.svg")
        dog_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg")

        # === Animation for Lecture Line 1 ===
        # Representing automatic filter learning with building asset
        self.place_at_grid(building_icon, 'B2', scale_factor=0.6)
        label_1 = Text("Layer 1: Edges", font_size=18, color=BLUE)
        self.place_at_grid(label_1, 'C2', scale_factor=0.9)
        
        self.play(FadeIn(building_icon), Write(label_1))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Representing stacking for features
        layer_2 = Square(side_length=1.2, color=GREEN).set_fill(GREEN, opacity=0.3)
        self.place_at_grid(layer_2, 'B4', scale_factor=1.0)
        label_2 = Text("Layer 2: Textures", font_size=18, color=GREEN)
        self.place_at_grid(label_2, 'C4', scale_factor=0.9)
        
        arrow1 = Arrow(building_icon.get_right(), layer_2.get_left(), buff=0.1, color=WHITE).scale(0.8)
        
        self.play(FadeIn(layer_2), GrowArrow(arrow1), Write(label_2))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Final recognition with dog asset
        self.place_at_grid(dog_icon, 'B6', scale_factor=0.5)
        label_3 = Text("Result: Dog", font_size=18, color=YELLOW)
        self.place_at_grid(label_3, 'C6', scale_factor=0.9)
        
        arrow2 = Arrow(layer_2.get_right(), dog_icon.get_left(), buff=0.1, color=WHITE).scale(0.8)
        
        self.play(FadeIn(dog_icon), GrowArrow(arrow2), Write(label_3))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
