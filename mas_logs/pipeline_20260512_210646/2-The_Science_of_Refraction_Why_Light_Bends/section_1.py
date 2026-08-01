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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the title and lecture lines as per prompt requirements
        self.setup_layout(
            "The Optical Illusion: The 'Broken' Pencil", 
            [
                "A fox sees a fish deep in the pond.", 
                "Light from the fish bends at the water's surface.", 
                "This creates an illusion, making the fish appear shallower."
            ]
        )
        
        # Define colors from Animation Description
        WATER_COLOR = "#1E90FF"
        FOX_COLOR = "#D2691E"
        FISH_COLOR = "#FFD700"
        RAY_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Water surface: horizontal line and a translucent rectangle area
        # Issue 33 & 54: Start water_rect from D2 to reduce crowding near text
        water_rect = Rectangle(width=4.5, height=3.0, fill_color=WATER_COLOR, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(water_rect, 'D2', 'F6')
        
        surface_line = Line(
            self.grid['D2'] + LEFT * 0.5, 
            self.grid['D6'] + RIGHT * 0.5, 
            color=WATER_COLOR, 
            stroke_width=4
        )
        
        # Issue 34 & 54: Fox at B6, scale 0.8
        fox = Text("🦊", font_size=48)
        self.place_at_grid(fox, 'B6', scale_factor=0.8)
        
        # Issue 35 & 54: Fish at E4, scale 0.8
        fish = Text("🐟", font_size=40)
        self.place_at_grid(fish, 'E4', scale_factor=0.8)
        
        self.play(
            FadeIn(water_rect),
            Create(surface_line),
            FadeIn(fox),
            FadeIn(fish),
            self.lecture[0].animate.set_color(WATER_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Light ray traveling from fish to surface and bending toward fox's eye
        # Intersection at D4 provides a vertical-to-angled bend (refraction away from normal)
        intersect_pt = self.grid['D4']
        
        # Ray from fish to surface
        ray_to_surface = Line(fish.get_center(), intersect_pt, color=RAY_COLOR)
        # Ray from surface to fox
        ray_to_fox = Line(intersect_pt, fox.get_center(), color=RAY_COLOR)
        
        self.play(
            Create(ray_to_surface),
            self.lecture[1].animate.set_color(RAY_COLOR)
        )
        self.play(Create(ray_to_fox))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Dashed ray extending straight back from the eye to a shallow, 'apparent' fish
        eye_pos = fox.get_center()
        dir_vec = intersect_pt - eye_pos
        
        # Apparent position is shallower than the real fish along the straight line of sight
        # Using a factor of 0.4 keeps it in the water but shallower
        apparent_pos = intersect_pt + dir_vec * 0.4
        
        dashed_ray = DashedLine(intersect_pt, apparent_pos, color=RAY_COLOR, stroke_opacity=0.5)
        
        # Apparent fish (translucent ghost fish)
        apparent_fish = Text("🐟", font_size=40).set_opacity(0.4)
        apparent_fish.move_to(apparent_pos)
        
        self.play(
            Create(dashed_ray),
            FadeIn(apparent_fish),
            self.lecture[2].animate.set_color(FISH_COLOR)
        )
        self.wait(2)
