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
        # Initializing the layout with title and lecture lines
        # Using exact strings and structure required by the TeachingScene base class
        self.setup_layout(
            "The Culinary Mystery: Root or Stem?", 
            [
                "It grows underground, so is a potato a root?", 
                "Meet Chef Spud, who is confused about his ingredients.", 
                "Let's investigate the biological identity of this popular veggie."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Show a cross-section of soil (#8B4513) with a potato (#D2B48C) and a carrot (#FFA500) positioned underground.
        
        # Soil cross-section: covers a significant portion of the right-side visual area (Rows C to F)
        soil = Rectangle(width=5.8, height=3.8, color="#8B4513", fill_opacity=0.8, stroke_width=0)
        self.place_in_area(soil, 'C1', 'F6')
        
        # Potato: Located in the soil at grid position D2
        potato = Ellipse(width=1.0, height=0.7, color="#D2B48C", fill_opacity=1)
        self.place_at_grid(potato, 'D2')
        
        # Carrot: Located in the soil at grid position D5
        # Represented as a triangle pointing downward to signify a taproot
        carrot = Triangle(color="#FFA500", fill_opacity=1)
        carrot.scale(0.5).rotate(PI) 
        self.place_at_grid(carrot, 'D5')
        
        self.play(
            FadeIn(soil),
            FadeIn(potato),
            FadeIn(carrot),
            self.lecture[0].animate.set_color("#D2B48C") # Highlight line with potato-related color
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A simple white chef hat (#FFFFFF) appears above the potato and tilts left and right to represent Chef Spud's confusion.
        
        # Chef Spud's hat: Placed at grid B2, which is visually above the potato at D2
        hat_base = Rectangle(width=0.6, height=0.25, color="#FFFFFF", fill_opacity=1)
        hat_puff = Circle(radius=0.3, color="#FFFFFF", fill_opacity=1).shift(UP * 0.15)
        chef_hat = VGroup(hat_base, hat_puff)
        self.place_at_grid(chef_hat, 'B2')
        
        self.play(
            FadeIn(chef_hat),
            self.lecture[1].animate.set_color("#FFFFFF") # Highlight line with hat color
        )
        
        # Representative confusion: Tilting left and right using rate_func=there_and_back
        self.play(Rotate(chef_hat, angle=0.2, about_point=chef_hat.get_bottom(), rate_func=there_and_back), run_time=0.6)
        self.play(Rotate(chef_hat, angle=-0.2, about_point=chef_hat.get_bottom(), rate_func=there_and_back), run_time=0.6)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A magnifying glass (#B0C4DE) moves over the potato as the soil fades out, leaving only the potato and carrot.
        
        # Magnifying glass construction (Lens and Handle)
        mg_circle = Circle(radius=0.45, color="#B0C4DE", stroke_width=8)
        mg_handle = Line(start=[0.35, -0.35, 0], end=[0.75, -0.75, 0], color="#B0C4DE", stroke_width=8)
        magnifying_glass = VGroup(mg_circle, mg_handle)
        
        # Starting the magnifying glass at an offset grid position (A4)
        self.place_at_grid(magnifying_glass, 'A4')
        
        # Soil and hat fade out while the magnifying glass moves to inspect the potato (at D2)
        self.play(
            FadeIn(magnifying_glass),
            FadeOut(soil),
            FadeOut(chef_hat),
            magnifying_glass.animate.move_to(self.grid['D2']),
            self.lecture[2].animate.set_color("#B0C4DE"), # Highlight line with magnifying glass color
            run_time=2
        )
        self.wait(2)
