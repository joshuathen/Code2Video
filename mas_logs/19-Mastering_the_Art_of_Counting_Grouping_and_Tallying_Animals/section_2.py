from manim import *
import os

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
        # Setup layout
        title_str = "Prerequisite Knowledge: Sorting by Attributes"
        lecture_lines_list = [
            "Let's group them by their kind!",
            "Lions go to the yellow box.",
            "Penguins and elephants find their homes too."
        ]
        self.setup_layout(title_str, lecture_lines_list)

        # Assets
        lion_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg"
        penguin_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/penguin.svg"
        elephant_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/elephant.svg"

        # Colors
        YELLOW_C = "#FFFF00"
        WHITE_C = "#FFFFFF"
        GREY_C = "#808080"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW_C))

        # Boxes
        lion_box = Rectangle(width=1.8, height=1.8, color=YELLOW_C, stroke_width=4)
        self.place_in_area(lion_box, "E1", "F2")
        
        penguin_box = Rectangle(width=1.8, height=1.8, color=WHITE_C, stroke_width=4)
        self.place_in_area(penguin_box, "E3", "F4")
        
        elephant_box = Rectangle(width=1.8, height=1.8, color=GREY_C, stroke_width=4)
        self.place_in_area(elephant_box, "E5", "F6")

        # Labels
        lion_label = Text("Lions", font_size=20, color=YELLOW_C)
        self.place_in_area(lion_label, "D1", "D2")
        
        penguin_label = Text("Penguins", font_size=20, color=WHITE_C)
        self.place_in_area(penguin_label, "D3", "D4")
        
        elephant_label = Text("Elephants", font_size=20, color=GREY_C)
        self.place_in_area(elephant_label, "D5", "D6")

        # Icons
        lions = VGroup(*[SVGMobject(lion_asset) for _ in range(3)])
        self.place_at_grid(lions[0], "A1", scale_factor=0.5)
        self.place_at_grid(lions[1], "B2", scale_factor=0.5)
        self.place_at_grid(lions[2], "C1", scale_factor=0.5)
        
        penguins = VGroup(*[SVGMobject(penguin_asset) for _ in range(3)])
        self.place_at_grid(penguins[0], "A3", scale_factor=0.5)
        self.place_at_grid(penguins[1], "B4", scale_factor=0.5)
        self.place_at_grid(penguins[2], "C3", scale_factor=0.5)
        
        elephants = VGroup(*[SVGMobject(elephant_asset) for _ in range(3)])
        self.place_at_grid(elephants[0], "A5", scale_factor=0.5)
        self.place_at_grid(elephants[1], "B6", scale_factor=0.5)
        self.place_at_grid(elephants[2], "C5", scale_factor=0.5)

        self.play(
            Create(lion_box), Create(penguin_box), Create(elephant_box),
            Write(lion_label), Write(penguin_label), Write(elephant_label),
            FadeIn(lions), FadeIn(penguins), FadeIn(elephants)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW_C)
        )
        
        self.play(
            lions[0].animate.move_to(self.grid["E1"]),
            lions[1].animate.move_to(self.grid["E2"]),
            lions[2].animate.move_to(self.grid["F1"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE_C)
        )
        
        self.play(
            penguins[0].animate.move_to(self.grid["E3"]),
            penguins[1].animate.move_to(self.grid["E4"]),
            penguins[2].animate.move_to(self.grid["F3"]),
            elephants[0].animate.move_to(self.grid["E5"]),
            elephants[1].animate.move_to(self.grid["E6"]),
            elephants[2].animate.move_to(self.grid["F5"]),
            run_time=2
        )
        self.wait(2)
