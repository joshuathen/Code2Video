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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application: The Heart of Neural Networks", 
                          ["Convolutions power neural networks.", 
                           "Machines learn optimal filter weights.", 
                           "Cameras identify traffic objects."])
        
        # Assets (Loaded once)
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        pedestrian_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pedestrian.svg")
        
        # Elements
        cnn_layer = VGroup(camera_icon, *[Circle(radius=0.3, color=BLUE, fill_opacity=0.5) for _ in range(4)])
        cnn_layer.arrange(DOWN)
        
        learned_weights = VGroup(*[Square(side_length=0.4, color=YELLOW, fill_opacity=0.5) for _ in range(3)])
        learned_weights.arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(cnn_layer, 'A4', 'C4', scale_factor=0.6)
        self.play(Create(cnn_layer))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(learned_weights, 'D5', 'D6', scale_factor=0.7)
        self.play(Create(learned_weights))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PURPLE)
        # Creating a node to flash, incorporating pedestrian asset
        pedestrian_icon.scale(0.5)
        node = VGroup(Dot(color=PURPLE, radius=0.2), pedestrian_icon)
        node.arrange(RIGHT)
        self.place_at_grid(node, 'E4', scale_factor=0.8)
        self.play(Flash(node[0], color="#FF00FF", line_length=0.2, num_lines=12, flash_radius=0.5))
        self.wait(1)
